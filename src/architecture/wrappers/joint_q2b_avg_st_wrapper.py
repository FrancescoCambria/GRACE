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
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

class CenterIntersection(nn.Module):
    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        # embeddings: (num_paths, N, dim)
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
        # embeddings: (num_paths, N, dim)
        outputs = F.relu(self.layer1(embeddings))
        gate = torch.sigmoid(self.layer2(outputs.mean(dim=0))) # (N, dim)
        offset, _ = torch.min(embeddings, dim=0) # (N, dim)
        return offset * gate

class Q2BAvgEncoder(nn.Module):
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
            
        self.center_intersection = CenterIntersection(hidden_dim)
        self.offset_intersection = BoxOffsetIntersection(hidden_dim)
        
        out_dim = learned_dim if learned_dim else 2 * hidden_dim
        self.projection = nn.Linear(2 * hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, batch_patterns, pattern_to_anchor_ids):
        batch_embs = []
        device = self.entity_embedding.weight.device
        
        for paths, original_str in batch_patterns:
            anchor_ids = pattern_to_anchor_ids.get(original_str)
            if anchor_ids is None or len(anchor_ids) == 0:
                batch_embs.append(torch.zeros(self.out_dim).to(device))
                continue
            
            anchor_ids = anchor_ids.to(device)
            centers = self.entity_embedding(anchor_ids)
            offsets = torch.zeros_like(centers)
            
            path_centers = []
            path_offsets = []
            
            for rel_ids in paths:
                curr_c = centers
                curr_o = offsets
                for r_id in rel_ids:
                    r_id_t = torch.tensor(r_id).to(device)
                    r_c = self.relation_embedding(r_id_t)
                    r_o = self.offset_embedding(r_id_t)
                    curr_c = curr_c + r_c
                    curr_o = curr_o + F.softplus(r_o)
                path_centers.append(curr_c)
                path_offsets.append(curr_o)
            
            if path_centers:
                if len(path_centers) > 1:
                    combined_c = self.center_intersection(torch.stack(path_centers))
                    combined_o = self.offset_intersection(torch.stack(path_offsets))
                else:
                    combined_c = path_centers[0]
                    combined_o = path_offsets[0]
                
                avg_c = combined_c.mean(dim=0)
                avg_o = combined_o.mean(dim=0)
                combined_emb = torch.cat([avg_c, avg_o], dim=-1)
            else:
                combined_emb = torch.zeros(2 * self.hidden_dim).to(device)
            
            batch_embs.append(self.projection(combined_emb))
            
        return torch.stack(batch_embs)

class JointSTQ2BAvgWDLModel(nn.Module):
    def __init__(self, st_model_name, q2b_params, st_learned_dim=None, dnn_hidden_units=(128, 128), device='cpu'):
        super().__init__()
        self.device = device
        self.st_model = SentenceTransformer(st_model_name).to(device)
        st_dim = self.st_model.get_sentence_embedding_dimension()
        self.st_out_dim = st_learned_dim if st_learned_dim else st_dim
        self.st_projection = nn.Linear(st_dim, self.st_out_dim) if st_learned_dim else nn.Identity()
        self.q2b_encoder = Q2BAvgEncoder(**q2b_params).to(device)
        total_dim = self.st_out_dim + 2 * self.q2b_encoder.out_dim
        self.feature_columns = [DenseFeat("feat", total_dim)]
        self.wdl_model = WDL(linear_feature_columns=self.feature_columns, dnn_feature_columns=self.feature_columns, dnn_hidden_units=dnn_hidden_units, task='binary', device=device)
        
    def forward(self, texts, body_patterns, head_patterns, pattern_to_anchor_ids):
        features = self.st_model.tokenize(texts)
        features = {k: v.to(self.device) for k, v in features.items()}
        st_emb = self.st_projection(self.st_model(features)['sentence_embedding'])
        body_emb = self.q2b_encoder(body_patterns, pattern_to_anchor_ids)
        head_emb = self.q2b_encoder(head_patterns, pattern_to_anchor_ids)
        combined = torch.cat([st_emb, body_emb, head_emb], dim=1)
        return self.wdl_model(combined)

class JointSTQ2BAvgDeepCTRWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, st_model_name='all-MiniLM-L6-v2', st_learned_dim=None, q2b_hidden_dim=64, q2b_learned_dim=64, entities_dict_path=None, relations_dict_path=None, checkpoint_path=None, dnn_hidden_units=(128, 128), epochs=5, batch_size=16, learning_rate=1e-5, device=None, neo4j_env_path='/home/cambria/MineGraphRule/GRAM/.env'):
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
        self.neo4j_env_path = neo4j_env_path
        self.model = None
        self.entity_dict = {}
        self.relation_dict = {}
        self.pattern_to_anchor_ids = {}
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

    def _get_anchors(self, patterns):
        unique_patterns = set(patterns)
        load_dotenv(self.neo4j_env_path)
        uri = os.getenv('NEO4J_URI', 'neo4j://localhost:37687')
        user = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'mineGraphRule')
        
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                for p_str in unique_patterns:
                    if not p_str or p_str in self.pattern_to_anchor_ids: continue
                    paths = [p.strip() for p in p_str.split(',')]
                    cypher_parts = []
                    anchor_label = None
                    for i, path in enumerate(paths):
                        c_path = path
                        if i == 0:
                            match = re.search(r'\((.*?)\)', c_path)
                            if match:
                                anchor_label = match.group(1)
                                c_path = c_path.replace(f'({anchor_label})', f'(n0:{anchor_label})', 1)
                        else:
                            match = re.search(r'\((.*?)\)', c_path)
                            if match:
                                label = match.group(1)
                                c_path = c_path.replace(f'({label})', f'(n0)', 1)
                        c_path = re.sub(r'\((?!n0)(.*?)\)', r'(:\1)', c_path)
                        c_path = re.sub(r'\[(.*?)\]', r'[:\1]', c_path)
                        cypher_parts.append(c_path)
                    query = f"MATCH {', '.join(cypher_parts)} RETURN DISTINCT n0.name as name LIMIT 1000"
                    res = session.run(query)
                    names = [rec['name'] for rec in res if rec['name'] in self.entity_dict]
                    if not names and anchor_label:
                        query = f"MATCH (n0:{anchor_label}) RETURN DISTINCT n0.name as name LIMIT 1000"
                        res = session.run(query)
                        names = [rec['name'] for rec in res if rec['name'] in self.entity_dict]
                    ids = [self.entity_dict[n] for n in names]
                    self.pattern_to_anchor_ids[p_str] = torch.tensor(ids, dtype=torch.long)
            driver.close()
        except Exception as e:
            print(f"Warning: Could not fetch anchors from Neo4j: {e}")

    def _parse_pattern_structure(self, p_str):
        if not isinstance(p_str, str): return ([], p_str)
        paths = p_str.split(',')
        res = []
        for ps in paths:
            rids = [self.relation_dict.get(rt) for rt in re.findall(r'-\[(.*?)\]->', ps) if self.relation_dict.get(rt) is not None]
            res.append(rids)
        return (res, p_str)

    def fit(self, X_text, X_body, X_body_names, X_head, X_head_names, y):
        self.classes_ = np.unique(y)
        self._get_anchors(list(X_body) + list(X_head))
        q2b_init = {}
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            state = torch.load(self.checkpoint_path, map_location='cpu')['model_state_dict']
            q2b_init = {'initial_entity_emb': state['entity_embedding'].cpu().numpy(), 'initial_relation_emb': state['relation_embedding'].cpu().numpy()}
            if 'offset_embedding' in state: q2b_init['initial_offset_emb'] = state['offset_embedding'].cpu().numpy()
        q2b_params = {'nentity': len(self.entity_dict) or 1000000, 'nrelation': len(self.relation_dict) or 100, 'hidden_dim': self.q2b_hidden_dim, 'learned_dim': self.q2b_learned_dim, **q2b_init}
        self.model = JointSTQ2BAvgWDLModel(self.st_model_name, q2b_params, self.st_learned_dim, self.dnn_hidden_units, self.device)
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        bp = [self._parse_pattern_structure(b) for b in X_body]
        hp = [self._parse_pattern_structure(h) for h in X_head]
        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X_text))
            for i in range(0, len(X_text), self.batch_size):
                idx = perm[i:i+self.batch_size]
                optimizer.zero_grad()
                out = self.model([X_text[j] for j in idx], [bp[j] for j in idx], [hp[j] for j in idx], self.pattern_to_anchor_ids).squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                loss = criterion(out, y_tensor[idx])
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X_text, X_body, X_body_names, X_head, X_head_names):
        self.model.eval()
        self._get_anchors(list(X_body) + list(X_head))
        bp = [self._parse_pattern_structure(b) for b in X_body]
        hp = [self._parse_pattern_structure(h) for h in X_head]
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_text), self.batch_size):
                out = self.model(X_text[i:i+self.batch_size], bp[i:i+self.batch_size], hp[i:i+self.batch_size], self.pattern_to_anchor_ids).squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                preds.extend((out > 0.5).cpu().numpy().astype(int))
        return np.array(preds)

    def predict_proba(self, X_text, X_body, X_body_names, X_head, X_head_names):
        self.model.eval()
        self._get_anchors(list(X_body) + list(X_head))
        bp = [self._parse_pattern_structure(b) for b in X_body]
        hp = [self._parse_pattern_structure(h) for h in X_head]
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_text), self.batch_size):
                out = self.model(X_text[i:i+self.batch_size], bp[i:i+self.batch_size], hp[i:i+self.batch_size], self.pattern_to_anchor_ids).squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                probs.extend(out.cpu().numpy())
        p1 = np.array(probs)
        return np.vstack([1-p1, p1]).T
