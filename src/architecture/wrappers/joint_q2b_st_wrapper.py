import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from deepctr_torch.models import WDL
from deepctr_torch.inputs import DenseFeat
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import re
import os

class CenterIntersection(nn.Module):
    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        # embeddings: (num_paths, dim)
        outputs = F.relu(self.layer1(embeddings))
        attention = F.softmax(self.layer2(outputs), dim=0)
        return torch.sum(attention * embeddings, dim=0)

class BoxOffsetIntersection(nn.Module):
    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        # embeddings: (num_paths, dim)
        outputs = F.relu(self.layer1(embeddings))
        gate = torch.sigmoid(self.layer2(outputs.mean(dim=0)))
        offset, _ = torch.min(embeddings, dim=0)
        return offset * gate

class Q2BEncoder(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, learned_dim=None, initial_entity_emb=None, initial_relation_emb=None, initial_offset_emb=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.entity_embedding = nn.Embedding(nentity, hidden_dim)
        self.relation_embedding = nn.Embedding(nrelation, hidden_dim)
        self.offset_embedding = nn.Embedding(nrelation, hidden_dim)
        
        if initial_entity_emb is not None:
            self.entity_embedding.weight.data.copy_(torch.from_numpy(initial_entity_emb))
        if initial_relation_emb is not None:
            self.relation_embedding.weight.data.copy_(torch.from_numpy(initial_relation_emb))
        if initial_offset_emb is not None:
            self.offset_embedding.weight.data.copy_(torch.from_numpy(initial_offset_emb))
            
        # Q2B Intersection Layers
        self.center_intersection = CenterIntersection(hidden_dim)
        self.offset_intersection = BoxOffsetIntersection(hidden_dim)
        
        out_dim = learned_dim if learned_dim else 2 * hidden_dim
        self.projection = nn.Linear(2 * hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, batch_patterns):
        batch_embs = []
        for paths in batch_patterns:
            path_centers = []
            path_offsets = []
            for start_node_id, rel_ids in paths:
                curr_c = self.entity_embedding(torch.tensor(start_node_id).to(self.entity_embedding.weight.device))
                curr_o = torch.zeros_like(curr_c)
                for r_id in rel_ids:
                    r_c = self.relation_embedding(torch.tensor(r_id).to(self.relation_embedding.weight.device))
                    r_o = self.offset_embedding(torch.tensor(r_id).to(self.offset_embedding.weight.device))
                    curr_c = curr_c + r_c
                    curr_o = curr_o + F.softplus(r_o)
                path_centers.append(curr_c)
                path_offsets.append(curr_o)
            
            if path_centers:
                if len(path_centers) > 1:
                    # USE Q2B INTERSECTION instead of mean
                    combined_c = self.center_intersection(torch.stack(path_centers))
                    combined_o = self.offset_intersection(torch.stack(path_offsets))
                else:
                    combined_c = path_centers[0]
                    combined_o = path_offsets[0]
                combined_emb = torch.cat([combined_c, combined_o], dim=-1)
            else:
                combined_emb = torch.zeros(2 * self.hidden_dim).to(self.entity_embedding.weight.device)
            
            batch_embs.append(combined_emb)
            
        return self.projection(torch.stack(batch_embs))

class JointSTQ2BWDLModel(nn.Module):
    def __init__(self, st_model_name, q2b_params, st_learned_dim=None, dnn_hidden_units=(128, 128), device='cpu'):
        super().__init__()
        self.device = device
        self.st_model = SentenceTransformer(st_model_name).to(device)
        st_dim = self.st_model.get_sentence_embedding_dimension()
        self.st_out_dim = st_learned_dim if st_learned_dim else st_dim
        self.st_projection = nn.Linear(st_dim, self.st_out_dim) if st_learned_dim else nn.Identity()
        self.q2b_encoder = Q2BEncoder(**q2b_params).to(device)
        total_dim = self.st_out_dim + 2 * self.q2b_encoder.out_dim
        self.feature_columns = [DenseFeat("feat", total_dim)]
        self.wdl_model = WDL(linear_feature_columns=self.feature_columns, dnn_feature_columns=self.feature_columns, dnn_hidden_units=dnn_hidden_units, task='binary', device=device)
        
    def forward(self, texts, body_patterns, head_patterns):
        features = self.st_model.tokenize(texts)
        features = {k: v.to(self.device) for k, v in features.items()}
        st_emb = self.st_projection(self.st_model(features)['sentence_embedding'])
        body_emb = self.q2b_encoder(body_patterns)
        head_emb = self.q2b_encoder(head_patterns)
        combined = torch.cat([st_emb, body_emb, head_emb], dim=1)
        return self.wdl_model(combined)

class JointSTQ2BDeepCTRWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, st_model_name='all-MiniLM-L6-v2', st_learned_dim=None, q2b_hidden_dim=64, q2b_learned_dim=64, entities_dict_path=None, relations_dict_path=None, checkpoint_path=None, dnn_hidden_units=(128, 128), epochs=5, batch_size=16, learning_rate=1e-5, device=None):
        self.st_model_name = st_model_name
        self.st_learned_dim = st_learned_dim
        self.q2b_hidden_dim = q2b_hidden_dim
        self.q2b_learned_dim = q2b_learned_dim
        self.entities_dict_path = entities_dict_path
        self.relations_dict_path = relations_dict_path
        self.checkpoint_path = checkpoint_path
        self.dnn_hidden_units = dnn_hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.entity_dict = {}
        self.relation_dict = {}
        self._load_dicts()

    def _load_dicts(self):
        if self.entities_dict_path and os.path.exists(self.entities_dict_path):
            with open(self.entities_dict_path, 'r') as f:
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 2: self.entity_dict[p[1]] = int(p[0])
        if self.relations_dict_path and os.path.exists(self.relations_dict_path):
            with open(self.relations_dict_path, 'r') as f:
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 2: self.relation_dict[p[1]] = int(p[0])

    def _parse_pattern(self, p_str, n_str):
        if not isinstance(p_str, str) or not isinstance(n_str, str): return []
        paths = p_str.split(',')
        names = [n.strip() for n in n_str.split(',') if n.strip()]
        res = []
        for i, ps in enumerate(paths):
            name = names[i] if i < len(names) else (names[0] if names else None)
            sid = self.entity_dict.get(name)
            if sid is None: continue
            rids = [self.relation_dict.get(rt) for rt in re.findall(r'-\[(.*?)\]->', ps) if self.relation_dict.get(rt) is not None]
            res.append((sid, rids))
        return res

    def fit(self, X_text, X_body, X_body_names, X_head, X_head_names, y):
        self.classes_ = np.unique(y)
        q2b_init = {}
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            state = torch.load(self.checkpoint_path, map_location='cpu')['model_state_dict']
            q2b_init = {'initial_entity_emb': state['entity_embedding'].cpu().numpy(), 'initial_relation_emb': state['relation_embedding'].cpu().numpy()}
            if 'offset_embedding' in state: q2b_init['initial_offset_emb'] = state['offset_embedding'].cpu().numpy()
        q2b_params = {'nentity': len(self.entity_dict) or 1000000, 'nrelation': len(self.relation_dict) or 100, 'hidden_dim': self.q2b_hidden_dim, 'learned_dim': self.q2b_learned_dim, **q2b_init}
        self.model = JointSTQ2BWDLModel(self.st_model_name, q2b_params, self.st_learned_dim, self.dnn_hidden_units, self.device)
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        bp = [self._parse_pattern(b, n) for b, n in zip(X_body, X_body_names)]
        hp = [self._parse_pattern(h, n) for h, n in zip(X_head, X_head_names)]
        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X_text))
            for i in range(0, len(X_text), self.batch_size):
                idx = perm[i:i+self.batch_size]
                optimizer.zero_grad()
                out = self.model([X_text[j] for j in idx], [bp[j] for j in idx], [hp[j] for j in idx]).squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                loss = criterion(out, y_t[idx])
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X_text, X_body, X_body_names, X_head, X_head_names):
        self.model.eval()
        bp = [self._parse_pattern(b, n) for b, n in zip(X_body, X_body_names)]
        hp = [self._parse_pattern(h, n) for h, n in zip(X_head, X_head_names)]
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_text), self.batch_size):
                out = self.model(X_text[i:i+self.batch_size], bp[i:i+self.batch_size], hp[i:i+self.batch_size]).squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                preds.extend((out > 0.5).cpu().numpy().astype(int))
        return np.array(preds)

    def predict_proba(self, X_text, X_body, X_body_names, X_head, X_head_names):
        self.model.eval()
        bp = [self._parse_pattern(b, n) for b, n in zip(X_body, X_body_names)]
        hp = [self._parse_pattern(h, n) for h, n in zip(X_head, X_head_names)]
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_text), self.batch_size):
                out = self.model(X_text[i:i+self.batch_size], bp[i:i+self.batch_size], hp[i:i+self.batch_size]).squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                probs.extend(out.cpu().numpy())
        p1 = np.array(probs)
        return np.vstack([1-p1, p1]).T
