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
import copy
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

class CenterIntersection(nn.Module):
    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        # embeddings: (num_paths, ..., dim)
        outputs = F.relu(self.layer1(embeddings))
        attention = F.softmax(self.layer2(outputs), dim=0)
        return torch.sum(attention * embeddings, dim=0)

class BoxOffsetIntersection(nn.Module):
    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2.weight.data.fill_(0) # Small initialization for gate
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        outputs = F.relu(self.layer1(embeddings))
        gate = torch.sigmoid(self.layer2(outputs.mean(dim=0)))
        offset, _ = torch.min(embeddings, dim=0)
        return offset * gate

class Q2BFlexibleEncoder(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, learned_dim=None, initial_entity_emb=None, initial_relation_emb=None, initial_offset_emb=None, mode="average"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mode = mode
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

    def forward(self, batch_data):
        batch_embs = []
        device = self.entity_embedding.weight.device
        
        for data in batch_data:
            if self.mode == "average":
                emb = self._forward_average(data, device)
            elif self.mode == "labelled":
                emb = self._forward_labelled(data, device)
            elif self.mode == "hybrid":
                emb = self._forward_hybrid(data, device)
            elif self.mode == "schema":
                emb = self._forward_schema(data, device)
            else:
                emb = torch.zeros(self.out_dim).to(device)
            batch_embs.append(emb)
            
        return torch.stack(batch_embs)

    def _forward_schema(self, data, device):
        """
        Schema mode: Operates on a KG where label nodes are the instances.
        Encodes rules based purely on logical paths between labels.
        """
        label_ids = data.get('label_ids', [])
        paths = data.get('paths', [])
        if not label_ids or label_ids[0] is None: return torch.zeros(self.out_dim).to(device)
        
        # Start at the label node
        c = self.entity_embedding(torch.tensor(label_ids[0]).to(device))
        o = torch.zeros_like(c)
        
        # Traverse the logical paths in the schema
        for rel_ids in paths:
            for r_id in rel_ids:
                if r_id is None: continue
                r_id_t = torch.tensor(r_id).to(device)
                c = c + self.relation_embedding(r_id_t)
                o = o + F.softplus(self.offset_embedding(r_id_t))
        
        if len(label_ids) > 1 and label_ids[-1] is not None:
            target_c = self.entity_embedding(torch.tensor(label_ids[-1]).to(device))
            target_o = torch.zeros_like(target_c)
            c = self.center_intersection(torch.stack([c, target_c]))
            o = self.offset_intersection(torch.stack([o, target_o]))
        
        combined = torch.cat([c, o], dim=-1)
        return self.projection(combined)

    def _forward_average(self, data, device):
        anchor_ids = data.get('anchors')
        paths = data.get('paths', [])
        if anchor_ids is None or len(anchor_ids) == 0:
            return torch.zeros(self.out_dim).to(device)
        
        anchor_ids = anchor_ids.to(device)
        centers = self.entity_embedding(anchor_ids)
        offsets = torch.zeros_like(centers)
        
        path_centers = []
        path_offsets = []
        
        for rel_ids in paths:
            curr_c, curr_o = centers, offsets
            for r_id in rel_ids:
                if r_id is None: continue
                r_id_t = torch.tensor(r_id).to(device)
                curr_c = curr_c + self.relation_embedding(r_id_t)
                curr_o = curr_o + F.softplus(self.offset_embedding(r_id_t))
            path_centers.append(curr_c)
            path_offsets.append(curr_o)
            
        if not path_centers: return torch.zeros(self.out_dim).to(device)
        
        c = self.center_intersection(torch.stack(path_centers)) if len(path_centers) > 1 else path_centers[0]
        o = self.offset_intersection(torch.stack(path_offsets)) if len(path_offsets) > 1 else path_offsets[0]
        
        combined = torch.cat([c.mean(dim=0), o.mean(dim=0)], dim=-1)
        return self.projection(combined)

    def _forward_labelled(self, data, device):
        label_ids = data.get('label_ids', [])
        paths = data.get('paths', [])
        if not label_ids or label_ids[0] is None: return torch.zeros(self.out_dim).to(device)
        
        c = self.entity_embedding(torch.tensor(label_ids[0]).to(device))
        o = torch.zeros_like(c)
        
        if paths and paths[0]:
            for r_id in paths[0]:
                if r_id is None: continue
                r_id_t = torch.tensor(r_id).to(device)
                c = c + self.relation_embedding(r_id_t)
                o = o + F.softplus(self.offset_embedding(r_id_t))
        
        if len(label_ids) > 1 and label_ids[-1] is not None:
            target_c = self.entity_embedding(torch.tensor(label_ids[-1]).to(device))
            target_o = torch.zeros_like(target_c)
            c = self.center_intersection(torch.stack([c, target_c]))
            o = self.offset_intersection(torch.stack([o, target_o]))
        
        combined = torch.cat([c, o], dim=-1)
        return self.projection(combined)

    def _forward_hybrid(self, data, device):
        target_instance_id = data.get('target_instance_id')
        if target_instance_id is None: return self._forward_labelled(data, device)
        
        label_ids = data.get('label_ids', [])
        paths = data.get('paths', [])
        if not label_ids or label_ids[0] is None: return torch.zeros(self.out_dim).to(device)
        
        c = self.entity_embedding(torch.tensor(label_ids[0]).to(device))
        o = torch.zeros_like(c)
        
        if paths and paths[0]:
            for r_id in paths[0]:
                if r_id is None: continue
                r_id_t = torch.tensor(r_id).to(device)
                c = c + self.relation_embedding(r_id_t)
                o = o + F.softplus(self.offset_embedding(r_id_t))
        
        target_c = self.entity_embedding(torch.tensor(target_instance_id).to(device))
        target_o = torch.zeros_like(target_c)
        
        c = self.center_intersection(torch.stack([c, target_c]))
        o = self.offset_intersection(torch.stack([o, target_o]))
        
        combined = torch.cat([c, o], dim=-1)
        return self.projection(combined)

class JointSTQ2BFlexibleModel(nn.Module):
    def __init__(self, st_model_name, q2b_params, use_st=True, st_learned_dim=None, dnn_hidden_units=(128, 128), device='cpu'):
        super().__init__()
        self.device = device
        self.use_st = use_st
        
        if use_st:
            self.st_model = SentenceTransformer(st_model_name).to(device)
            st_dim = self.st_model.get_sentence_embedding_dimension()
            self.st_out_dim = st_learned_dim if st_learned_dim else st_dim
            self.st_projection = nn.Linear(st_dim, self.st_out_dim) if st_learned_dim else nn.Identity()
        else:
            self.st_out_dim = 0
            
        self.q2b_encoder = Q2BFlexibleEncoder(**q2b_params).to(device)
        total_dim = self.st_out_dim + 2 * self.q2b_encoder.out_dim
        self.feature_columns = [DenseFeat("feat", total_dim)]
        self.wdl_model = WDL(linear_feature_columns=self.feature_columns, dnn_feature_columns=self.feature_columns, dnn_hidden_units=dnn_hidden_units, task='binary', device=device)
        
    def forward(self, texts, body_data, head_data):
        body_emb = self.q2b_encoder(body_data)
        head_emb = self.q2b_encoder(head_data)

        if self.use_st:
            # Ensure texts is a list of strings for SentenceTransformer
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            features = self.st_model.tokenize(texts)
            features = {k: v.to(self.device) for k, v in features.items()}
            st_emb = self.st_projection(self.st_model(features)['sentence_embedding'])
            combined = torch.cat([st_emb, body_emb, head_emb], dim=1)
        else:
            combined = torch.cat([body_emb, head_emb], dim=1)

        return self.wdl_model(combined)

class JointSTQ2BFlexibleWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, mode="average", use_st=True, st_model_name='all-MiniLM-L6-v2', st_learned_dim=None, q2b_hidden_dim=64, q2b_learned_dim=64, entities_dict_path=None, relations_dict_path=None, checkpoint_path=None, dnn_hidden_units=(128, 128), epochs=5, batch_size=16, learning_rate=1e-5, device=None, neo4j_env_path='/home/cambria/MineGraphRule/GRAM/.env', early_stopping_patience=10, use_lr_scheduler=False):
        self.mode = mode
        self.use_st = use_st
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
        self.early_stopping_patience = early_stopping_patience
        self.use_lr_scheduler = use_lr_scheduler
        self.model = None
        self.entity_dict = {}
        self.relation_dict = {}
        self.pattern_to_anchors = {}
        self.history_ = {'train_loss': [], 'val_loss': [], 'lr': []}
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
        if self.mode != "average": return
        load_dotenv(self.neo4j_env_path)
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        pw = os.getenv('NEO4J_PASSWORD')
        try:
            from neo4j import basic_auth
            driver = GraphDatabase.driver(uri, auth=basic_auth(user, pw))
            with driver.session() as session:
                for p_str in set(patterns):
                    if not p_str or p_str in self.pattern_to_anchors: continue
                    labels = re.findall(r'\((.*?)\)', p_str)
                    if not labels: continue
                    query = f"MATCH (n0:{labels[0]}) RETURN DISTINCT n0.name as name"
                    res = session.run(query)
                    ids = [self.entity_dict[r['name']] for r in res if r['name'] in self.entity_dict]
                    self.pattern_to_anchors[p_str] = torch.tensor(ids, dtype=torch.long)
            driver.close()
        except Exception as e: print(f"Neo4j Error: {e}")

    def _parse(self, p_str, target_name=None):
        rels = []
        labels = []
        if isinstance(p_str, str):
            labels = re.findall(r'\((.*?)\)', p_str)
            for ps in p_str.split(','):
                rels.append([self.relation_dict.get(r) for r in re.findall(r'-\[(.*?)\]->', ps) if self.relation_dict.get(r) is not None])
        return {
            'paths': rels,
            'labels': labels,
            'label_ids': [self.entity_dict.get(l) for l in labels],
            'anchors': self.pattern_to_anchors.get(p_str),
            'target_instance_id': self.entity_dict.get(target_name)
        }

    def fit(self, X_text, X_body, X_body_names, X_head, X_head_names, y, validation_split=0.2):
        self._get_anchors(list(X_body) + list(X_head))
        q2b_init = {}
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            state = torch.load(self.checkpoint_path, map_location='cpu')['model_state_dict']
            if 'entity_embedding' in state: q2b_init['initial_entity_emb'] = state['entity_embedding'].cpu().numpy()
            if 'relation_embedding' in state: q2b_init['initial_relation_emb'] = state['relation_embedding'].cpu().numpy()
            if 'offset_embedding' in state: q2b_init['initial_offset_emb'] = state['offset_embedding'].cpu().numpy()
        
        nentity = len(self.entity_dict)
        if nentity == 0 and 'initial_entity_emb' in q2b_init:
            nentity = q2b_init['initial_entity_emb'].shape[0]
            print(f"[WARNING] entity_dict is empty, using nentity={nentity} from loaded checkpoint.")
        elif nentity == 0:
            nentity = 1000000

        q2b_params = {'nentity': nentity, 'nrelation': len(self.relation_dict) or 100, 'hidden_dim': self.q2b_hidden_dim, 'learned_dim': self.q2b_learned_dim, 'mode': self.mode, **q2b_init}
        self.model = JointSTQ2BFlexibleModel(self.st_model_name, q2b_params, use_st=self.use_st, st_learned_dim=self.st_learned_dim, dnn_hidden_units=self.dnn_hidden_units, device=self.device)
        self.model.to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Adaptive LR Scheduler
        scheduler = None
        if self.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        indices = np.arange(len(X_text))
        it, iv = train_test_split(indices, test_size=validation_split, random_state=42, stratify=y)
        
        y_t = torch.tensor(y[it], dtype=torch.float32).to(self.device)
        y_v = torch.tensor(y[iv], dtype=torch.float32).to(self.device)
        
        bd_t = [self._parse(X_body[j], X_body_names[j]) for j in it]
        hd_t = [self._parse(X_head[j], X_head_names[j]) for j in it]
        bd_v = [self._parse(X_body[j], X_body_names[j]) for j in iv]
        hd_v = [self._parse(X_head[j], X_head_names[j]) for j in iv]
        
        self.history_ = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0
            perm = torch.randperm(len(it))
            for i in range(0, len(it), self.batch_size):
                idx = perm[i:i+self.batch_size]
                optimizer.zero_grad()
                out = self.model([X_text[it[j]] for j in idx], [bd_t[j] for j in idx], [hd_t[j] for j in idx]).squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                loss = criterion(out, y_t[idx])
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(idx)
            
            avg_train_loss = total_train_loss / len(it)
            self.history_['train_loss'].append(avg_train_loss)
            
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for i in range(0, len(iv), self.batch_size):
                    end = min(i + self.batch_size, len(iv))
                    out = self.model([X_text[iv[j]] for j in range(i, end)], bd_v[i:end], hd_v[i:end]).squeeze()
                    if out.dim() == 0: out = out.unsqueeze(0)
                    loss = criterion(out, y_v[i:end])
                    total_val_loss += loss.item() * (end - i)
            
            avg_val_loss = total_val_loss / len(iv)
            self.history_['val_loss'].append(avg_val_loss)
            
            curr_lr = optimizer.param_groups[0]['lr']
            self.history_['lr'].append(curr_lr)
            
            print(f"Epoch {epoch+1}/{self.epochs} - train_loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f} - lr: {curr_lr:.2e}")
            
            # Step scheduler
            if scheduler:
                scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Restoring best model weights.")
                    self.model.load_state_dict(best_model_state)
                    break
                    
        return self

    def predict_proba(self, X_text, X_body, X_body_names, X_head, X_head_names):
        self.model.eval()
        self._get_anchors(list(X_body) + list(X_head))
        bd = [self._parse(b, n) for b, n in zip(X_body, X_body_names)]
        hd = [self._parse(h, n) for h, n in zip(X_head, X_head_names)]
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_text), self.batch_size):
                out = self.model(X_text[i:i+self.batch_size], bd[i:i+self.batch_size], hd[i:i+self.batch_size]).squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                probs.extend(out.cpu().numpy())
        p1 = np.array(probs)
        return np.vstack([1-p1, p1]).T

    def predict(self, X_text, X_body, X_body_names, X_head, X_head_names):
        probs = self.predict_proba(X_text, X_body, X_body_names, X_head, X_head_names)
        return (probs[:, 1] > 0.5).astype(int)
