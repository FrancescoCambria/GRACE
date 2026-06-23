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
import csv
import io
import pickle
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

class RotatEEncoder(nn.Module):
    def __init__(self, nentity, hidden_dim, learned_dim=None, initial_entity_emb=None):
        super().__init__()
        # RotatE entity embeddings are 2 * hidden_dim
        self.entity_embedding = nn.Embedding(nentity, 2 * hidden_dim)
        
        if initial_entity_emb is not None:
            self.entity_embedding.weight.data.copy_(torch.from_numpy(initial_entity_emb))
            
        self.hidden_dim = hidden_dim
        out_dim = learned_dim if learned_dim else 2 * hidden_dim
        self.projection = nn.Linear(2 * hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, embeddings):
        # embeddings is (batch, 2 * hidden_dim)
        return self.projection(embeddings)

class JointSTRotatEModel(nn.Module):
    def __init__(self, st_model_name, rotate_params, use_st=True, st_learned_dim=None, use_metrics=False, metric_dim=16, dnn_hidden_units=(128, 128), device='cpu'):
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
            
        self.rotate_encoder = RotatEEncoder(**rotate_params).to(device)
        
        self.use_metrics = use_metrics
        self.metric_out_dim = metric_dim if use_metrics else 0
        if use_metrics:
            self.half_dim = metric_dim // 2
            self.other_half_dim = metric_dim - self.half_dim
            
            self.support_proj = nn.Sequential(
                nn.Linear(1, self.half_dim),
                nn.ReLU(),
                nn.Linear(self.half_dim, self.half_dim)
            ).to(device)
            
            self.confidence_proj = nn.Sequential(
                nn.Linear(1, self.other_half_dim),
                nn.ReLU(),
                nn.Linear(self.other_half_dim, self.other_half_dim)
            ).to(device)
        
        # We have body embedding and head embedding
        total_dim = self.st_out_dim + 2 * self.rotate_encoder.out_dim + self.metric_out_dim
        self.feature_columns = [DenseFeat("feat", total_dim)]
        self.wdl_model = WDL(linear_feature_columns=self.feature_columns, dnn_feature_columns=self.feature_columns, dnn_hidden_units=dnn_hidden_units, task='binary', device=device)
        
    def forward(self, texts, body_rotate_embs, head_rotate_embs, metrics=None):
        body_emb = self.rotate_encoder(body_rotate_embs)
        head_emb = self.rotate_encoder(head_rotate_embs)
        
        if self.use_st:
            # Ensure texts is a list of strings for SentenceTransformer
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            features = self.st_model.tokenize(texts)
            features = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
            st_emb = self.st_projection(self.st_model(features)['sentence_embedding'])
            parts = [st_emb, body_emb, head_emb]
        else:
            parts = [body_emb, head_emb]
            
        if self.use_metrics and metrics is not None:
            support = metrics[:, 0].unsqueeze(1)
            confidence = metrics[:, 1].unsqueeze(1)
            supp_feats = self.support_proj(support)
            conf_feats = self.confidence_proj(confidence)
            metric_feats = torch.cat([supp_feats, conf_feats], dim=1)
            parts.append(metric_feats)
            
        combined = torch.cat(parts, dim=1)
        return self.wdl_model(combined)

class JointSTRotatEWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, use_st=True, st_model_name='all-MiniLM-L6-v2', st_learned_dim=None, rotate_hidden_dim=192, rotate_learned_dim=64, use_metrics=False, metric_dim=16, entities_dict_path=None, checkpoint_path=None, dnn_hidden_units=(128, 128), epochs=5, batch_size=16, learning_rate=1e-5, device=None, neo4j_env_path='/home/cambria/gram3/ClassificationforMineGraphRule/.env', early_stopping_patience=10, use_lr_scheduler=False, use_instances=False, cache_path='kge/pattern_embeddings_cache.pkl'):
        self.use_metrics = use_metrics
        self.metric_dim = metric_dim
        self.use_st = use_st
        self.st_model_name = st_model_name
        self.st_learned_dim = st_learned_dim
        self.rotate_hidden_dim = rotate_hidden_dim
        self.rotate_learned_dim = rotate_learned_dim
        self.entities_dict_path = entities_dict_path
        self.checkpoint_path = checkpoint_path
        self.dnn_hidden_units = dnn_hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.neo4j_env_path = neo4j_env_path
        self.early_stopping_patience = early_stopping_patience
        self.use_lr_scheduler = use_lr_scheduler
        self.use_instances = use_instances
        self.cache_path = cache_path
        
        self.model = None
        self.entity_dict = {}
        self.entity_emb_matrix = None
        self.pattern_cache = {}
        self.history_ = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        self._load_dicts_and_embeddings()
        self._load_persistent_cache()

    def _load_dicts_and_embeddings(self):
        if self.entities_dict_path and os.path.exists(self.entities_dict_path):
            with open(self.entities_dict_path, 'r') as f:
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 2: self.entity_dict[p[1]] = int(p[0])
        
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            state = torch.load(self.checkpoint_path, map_location='cpu')['model_state_dict']
            if 'entity_embedding' in state:
                self.entity_emb_matrix = state['entity_embedding'].cpu().numpy()
                print(f"Loaded entity embeddings from {self.checkpoint_path}, shape: {self.entity_emb_matrix.shape}")

    def _load_persistent_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    self.pattern_cache = pickle.load(f)
                print(f"Loaded {len(self.pattern_cache)} entries from persistent cache: {self.cache_path}")
            except Exception as e:
                print(f"Error loading persistent cache: {e}")
                self.pattern_cache = {}
        else:
            self.pattern_cache = {}

    def _save_persistent_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.pattern_cache, f)
            print(f"Saved {len(self.pattern_cache)} entries to persistent cache: {self.cache_path}")
        except Exception as e:
            print(f"Error saving persistent cache: {e}")

    def _get_pattern_embedding(self, pattern_str, names_str, anchor_label, session):
        # Cache check
        cache_key = (pattern_str, names_str) if self.use_instances else pattern_str
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]

        parts = [p.strip() for p in pattern_str.split(',')]
        
        if isinstance(names_str, str):
            f = io.StringIO(names_str)
            reader = csv.reader(f, delimiter=',', quotechar='"')
            try:
                names_list = next(reader, [])
            except StopIteration:
                names_list = []
        else:
            names_list = []
        
        path_embeddings = []
        
        for i, part in enumerate(parts):
            target_name = names_list[i].strip() if i < len(names_list) and self.use_instances else None
            labels = re.findall(r'\((.*?)\)', part)
            rels = re.findall(r'-\[(.*?)\]->', part)
            
            if not labels: continue
            
            # Neo4j Query Construction
            match_clause = "MATCH p="
            node_clauses = []
            where_clauses = []
            for j, label in enumerate(labels):
                if self.use_instances and j == len(labels) - 1 and target_name:
                    if target_name.isdigit():
                        node_clauses.append(f"(n{j}:{label})")
                        where_clauses.append(f"id(n{j}) = {target_name}")
                    else:
                        safe_name = target_name.replace("'", "\\'")
                        node_clauses.append(f"(n{j}:{label} {{name: '{safe_name}'}})")
                else:
                    node_clauses.append(f"(n{j}:{label})")
            
            for j in range(len(rels)):
                match_clause += node_clauses[j] + f"-[:{rels[j]}]->"
            match_clause += node_clauses[-1]
            
            query = f"{match_clause}"
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            query += " RETURN p LIMIT 50"
            
            instance_embeddings = []
            try:
                result = session.run(query)
                for record in result:
                    path = record["p"]
                    nodes = list(path.nodes)
                    node_vectors = []
                    for node in nodes:
                        potential_keys = []
                        if "name" in node: potential_keys.append(node["name"])
                        if "title" in node: potential_keys.append(node["title"])
                        potential_keys.append(str(node.id))
                        
                        for key in potential_keys:
                            if key in self.entity_dict:
                                node_vectors.append(self.entity_emb_matrix[self.entity_dict[key]])
                                break
                    
                    if node_vectors:
                        # Average embeddings WITHIN this path instance
                        instance_embeddings.append(np.mean(node_vectors, axis=0))
            except Exception:
                pass
            
            if instance_embeddings:
                # Average embeddings ACROSS all instances of this part of the pattern
                path_embeddings.append(np.mean(instance_embeddings, axis=0))
            else:
                # Fallback to zeros if no instance found
                path_embeddings.append(np.zeros(2 * self.rotate_hidden_dim))
        
        # Average results from all pattern parts (if multiple)
        if path_embeddings:
            final_emb = np.mean(path_embeddings, axis=0)
        else:
            final_emb = np.zeros(2 * self.rotate_hidden_dim)
            
        self.pattern_cache[cache_key] = final_emb
        return final_emb

    def _prepare_data(self, X_body, X_body_names, X_head, X_head_names, anchor_labels):
        if os.path.exists(self.neo4j_env_path):
            load_dotenv(self.neo4j_env_path)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            load_dotenv(os.path.join(base_dir, '.env'))
            
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        pw = os.getenv('NEO4J_PASSWORD')
        
        body_embs = []
        head_embs = []
        
        # Track if we actually performed any NEW queries
        initial_cache_size = len(self.pattern_cache)

        print(f"Connecting to Neo4j to fetch pattern instances...")
        try:
            from neo4j import basic_auth
            auth = basic_auth(user or "", pw or "")
            driver = GraphDatabase.driver(uri, auth=auth)
            # Test connection
            driver.verify_connectivity()
            
            with driver.session() as session:
                for i in range(len(X_body)):
                    if i % 100 == 0:
                        print(f"Processing rule {i}/{len(X_body)}...")
                    body_emb = self._get_pattern_embedding(X_body[i], X_body_names[i], anchor_labels[i], session)
                    head_emb = self._get_pattern_embedding(X_head[i], X_head_names[i], anchor_labels[i], session)
                    body_embs.append(body_emb)
                    head_embs.append(head_emb)
            driver.close()
        except Exception as e:
            print(f"\n[CRITICAL WARNING] Neo4j connection error: {e}")
            print("[CRITICAL WARNING] Falling back to zero embeddings for all patterns. This model will likely NOT learn correctly.")
            # Fallback to zeros
            body_embs = [np.zeros(2 * self.rotate_hidden_dim) for _ in range(len(X_body))]
            head_embs = [np.zeros(2 * self.rotate_hidden_dim) for _ in range(len(X_body))]
        
        # Save cache if it has grown
        if len(self.pattern_cache) > initial_cache_size:
            self._save_persistent_cache()
            
        return torch.tensor(np.array(body_embs), dtype=torch.float32), torch.tensor(np.array(head_embs), dtype=torch.float32)

    def fit(self, X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels, y, X_metrics=None, validation_split=0.2):
        print("Preprocessing patterns with Neo4j and RotatE embeddings...")
        body_rotate_embs, head_rotate_embs = self._prepare_data(X_body, X_body_names, X_head, X_head_names, anchor_labels)
        
        rotate_params = {
            'nentity': len(self.entity_dict) or 1000000,
            'hidden_dim': self.rotate_hidden_dim,
            'learned_dim': self.rotate_learned_dim,
            'initial_entity_emb': self.entity_emb_matrix
        }
        
        self.model = JointSTRotatEModel(
            self.st_model_name, 
            rotate_params, 
            use_st=self.use_st, 
            st_learned_dim=self.st_learned_dim, 
            use_metrics=getattr(self, 'use_metrics', False),
            metric_dim=getattr(self, 'metric_dim', 16),
            dnn_hidden_units=self.dnn_hidden_units, 
            device=self.device
        )
        self.model.to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Adaptive LR Scheduler
        scheduler = None
        if self.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        
        indices = np.arange(len(X_text))
        
        actual_val_split = validation_split
        if len(indices) < 5:
            actual_val_split = 0.2
            
        try:
            it, iv = train_test_split(indices, test_size=actual_val_split, random_state=42, stratify=y)
        except ValueError:
            it, iv = train_test_split(indices, test_size=actual_val_split, random_state=42)
            
        if len(it) == 0:
            it, iv = indices[:1], indices[1:]
        
        y_t = torch.tensor(y[it], dtype=torch.float32).to(self.device)
        y_v = torch.tensor(y[iv], dtype=torch.float32).to(self.device)
        
        body_rotate_embs_t = body_rotate_embs[it].to(self.device)
        head_rotate_embs_t = head_rotate_embs[it].to(self.device)
        body_rotate_embs_v = body_rotate_embs[iv].to(self.device)
        head_rotate_embs_v = head_rotate_embs[iv].to(self.device)
        
        if getattr(self, 'use_metrics', False) and X_metrics is not None:
            metrics_t = torch.tensor(X_metrics[it], dtype=torch.float32).to(self.device)
            metrics_v = torch.tensor(X_metrics[iv], dtype=torch.float32).to(self.device)
        else:
            metrics_t, metrics_v = None, None
            
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
                out = self.model(
                    [X_text[it[j]] for j in idx], 
                    body_rotate_embs_t[idx], 
                    head_rotate_embs_t[idx],
                    metrics=metrics_t[idx] if metrics_t is not None else None
                ).squeeze()
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
                    out = self.model(
                        [X_text[iv[j]] for j in range(i, end)], 
                        body_rotate_embs_v[i:end], 
                        head_rotate_embs_v[i:end],
                        metrics=metrics_v[i:end] if metrics_v is not None else None
                    ).squeeze()
                    if out.dim() == 0: out = out.unsqueeze(0)
                    loss = criterion(out, y_v[i:end])
                    total_val_loss += loss.item() * (end - i)
            
            avg_val_loss = total_val_loss / len(iv)
            self.history_['val_loss'].append(avg_val_loss)
            
            curr_lr = optimizer.param_groups[0]['lr']
            self.history_['lr'].append(curr_lr)
            
            print(f"Epoch {epoch+1}/{self.epochs} - train_loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f} - lr: {curr_lr:.2e}")
            
            if scheduler:
                scheduler.step(avg_val_loss)
            
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

    def predict_proba(self, X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels, X_metrics=None):
        self.model.eval()
        body_rotate_embs, head_rotate_embs = self._prepare_data(X_body, X_body_names, X_head, X_head_names, anchor_labels)
        body_rotate_embs = body_rotate_embs.to(self.device)
        head_rotate_embs = head_rotate_embs.to(self.device)
        
        if getattr(self, 'use_metrics', False) and X_metrics is not None:
            metrics_tensor = torch.tensor(X_metrics, dtype=torch.float32).to(self.device)
        else:
            metrics_tensor = None
            
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_text), self.batch_size):
                end = min(i + self.batch_size, len(X_text))
                out = self.model(
                    X_text[i:end], 
                    body_rotate_embs[i:end], 
                    head_rotate_embs[i:end],
                    metrics=metrics_tensor[i:end] if metrics_tensor is not None else None
                ).squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                probs.extend(out.cpu().numpy())
        p1 = np.array(probs)
        return np.vstack([1-p1, p1]).T

    def predict(self, X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels, X_metrics=None):
        probs = self.predict_proba(X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels, X_metrics=X_metrics)
        return (probs[:, 1] > 0.5).astype(int)
