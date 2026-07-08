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
    def __init__(self, hidden_dim, learned_dim=None, **kwargs):
        super().__init__()
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
    def __init__(self, use_st=True, st_model_name='all-MiniLM-L6-v2', st_learned_dim=None, rotate_hidden_dim=192, rotate_learned_dim=64, use_metrics=False, metric_dim=16, entities_dict_path=None, relations_dict_path=None, checkpoint_path=None, dnn_hidden_units=(128, 128), epochs=5, batch_size=16, learning_rate=1e-5, device=None, neo4j_env_path='/home/cambria/gram3/ClassificationforMineGraphRule/.env', early_stopping_patience=10, use_lr_scheduler=False, use_instances=False, cache_path='kge/pattern_structure_cache.pkl'):
        self.use_metrics = use_metrics
        self.metric_dim = metric_dim
        self.use_st = use_st
        self.st_model_name = st_model_name
        self.st_learned_dim = st_learned_dim
        self.rotate_hidden_dim = rotate_hidden_dim
        self.rotate_learned_dim = rotate_learned_dim
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
        self.use_instances = use_instances
        self.cache_path = cache_path
        
        self.model = None
        self.entity_dict = {}
        self.relation_dict = {}
        self.entity_emb_matrix = None
        self.relation_emb_matrix = None
        self.embedding_range = 24.0
        self.pattern_cache = {}
        self.history_ = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        # Automatically resolve relations_dict_path if not provided
        if not self.relations_dict_path and self.entities_dict_path:
            self.relations_dict_path = self.entities_dict_path.replace("entities.dict", "relations.dict")
            
        self._load_dicts_and_embeddings()
        self._load_persistent_cache()

    def _load_dicts_and_embeddings(self):
        if self.entities_dict_path and os.path.exists(self.entities_dict_path):
            with open(self.entities_dict_path, 'r') as f:
                for line in f:
                    p = line.strip().split('	')
                    if len(p) >= 2: self.entity_dict[p[1]] = int(p[0])
                    
        if self.relations_dict_path and os.path.exists(self.relations_dict_path):
            with open(self.relations_dict_path, 'r') as f:
                for line in f:
                    p = line.strip().split('	')
                    if len(p) >= 2: self.relation_dict[p[1]] = int(p[0])
        
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            state = checkpoint['model_state_dict']
            if 'entity_embedding' in state:
                self.entity_emb_matrix = state['entity_embedding'].cpu().numpy()
                print(f"Loaded entity embeddings from {self.checkpoint_path}, shape: {self.entity_emb_matrix.shape}")
            if 'relation_embedding' in state:
                self.relation_emb_matrix = state['relation_embedding'].cpu().numpy()
                print(f"Loaded relation embeddings from {self.checkpoint_path}, shape: {self.relation_emb_matrix.shape}")
            if 'embedding_range' in state:
                self.embedding_range = state['embedding_range'].item()
            elif 'embedding_range' in checkpoint:
                self.embedding_range = checkpoint['embedding_range']

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

    def _get_pattern_structure(self, pattern_str, names_str, anchor_label, session):
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
        
        all_paths_structure = []
        
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
                        safe_name = target_name.replace("'", "\'")
                        node_clauses.append(f"(n{j}:{label} {{name: '{safe_name}'}})")
                else:
                    node_clauses.append(f"(n{j}:{label})")
            
            for j in range(len(rels)):
                match_clause += node_clauses[j] + f"-[:{rels[j]}]->"
            match_clause += node_clauses[-1]
            
            query = f"{match_clause}"
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            query += " RETURN p"
            
            try:
                result = session.run(query)
                for record in result:
                    path = record["p"]
                    nodes = list(path.nodes)
                    relationships = list(path.relationships)
                    
                    if not nodes: continue
                    
                    # Start node KGE ID
                    start_node = nodes[0]
                    start_key = None
                    potential_keys = []
                    if "name" in start_node: potential_keys.append(start_node["name"])
                    if "title" in start_node: potential_keys.append(start_node["title"])
                    potential_keys.append(str(start_node.id))
                    
                    for key in potential_keys:
                        if key in self.entity_dict:
                            start_key = key
                            break
                            
                    if start_key is None: continue
                    start_id = self.entity_dict[start_key]
                    
                    # Relation KGE IDs
                    rel_ids = []
                    for rel in relationships:
                        if rel.type in self.relation_dict:
                            rel_ids.append(self.relation_dict[rel.type])
                            
                    all_paths_structure.append((start_id, rel_ids))
            except Exception:
                pass
                
        self.pattern_cache[cache_key] = all_paths_structure
        return all_paths_structure

    def _compute_embeddings_from_structure(self, paths_structure):
        embs = []
        for paths in paths_structure:
            if not paths:
                embs.append(np.zeros(2 * self.rotate_hidden_dim))
                continue
                
            path_vectors = []
            for start_ent_id, rel_ids in paths:
                if self.entity_emb_matrix is None:
                    curr_emb = np.zeros(2 * self.rotate_hidden_dim)
                else:
                    curr_emb = self.entity_emb_matrix[start_ent_id]
                
                curr_x = curr_emb[:self.rotate_hidden_dim]
                curr_y = curr_emb[self.rotate_hidden_dim:]
                
                for rel_id in rel_ids:
                    if self.relation_emb_matrix is not None:
                        rel_emb = self.relation_emb_matrix[rel_id]
                        pi = 3.14159265358979323846
                        phase = rel_emb / (self.embedding_range / pi)
                        cos_r = np.cos(phase)
                        sin_r = np.sin(phase)
                        
                        x_new = curr_x * cos_r - curr_y * sin_r
                        y_new = curr_x * sin_r + curr_y * cos_r
                        curr_x, curr_y = x_new, y_new
                        
                path_vectors.append(np.concatenate([curr_x, curr_y]))
                
            embs.append(np.mean(path_vectors, axis=0))
        return torch.tensor(np.array(embs), dtype=torch.float32)

    def _update_kge_embeddings(self, body_grad, head_grad, body_paths, head_paths):
        entity_grads = {}
        
        def accumulate(grad_tensor, paths_list):
            for rule_idx, rule_grad in enumerate(grad_tensor):
                paths = paths_list[rule_idx]
                if not paths: continue
                num_paths = len(paths)
                
                for start_ent_id, rel_ids in paths:
                    g_x = rule_grad[:self.rotate_hidden_dim]
                    g_y = rule_grad[self.rotate_hidden_dim:]
                    
                    # Compute cumulative rotation along the path
                    cos_cum, sin_cum = 1.0, 0.0
                    for rel_id in rel_ids:
                        if self.relation_emb_matrix is not None:
                            rel_emb = self.relation_emb_matrix[rel_id]
                            pi = 3.14159265358979323846
                            phase = rel_emb / (self.embedding_range / pi)
                            cos_r = np.cos(phase)
                            sin_r = np.sin(phase)
                            
                            cos_new = cos_cum * cos_r - sin_cum * sin_r
                            sin_new = sin_cum * cos_r + cos_cum * sin_r
                            cos_cum, sin_cum = cos_new, sin_new
                            
                    # Apply inverse rotation to backpropagate to start node
                    g_start_x = (g_x * cos_cum + g_y * sin_cum) / num_paths
                    g_start_y = (-g_x * sin_cum + g_y * cos_cum) / num_paths
                    g_start = np.concatenate([g_start_x, g_start_y])
                    
                    if start_ent_id not in entity_grads:
                        entity_grads[start_ent_id] = np.zeros_like(g_start)
                    entity_grads[start_ent_id] += g_start
                    
        accumulate(body_grad, body_paths)
        accumulate(head_grad, head_paths)
        
        # Apply manual SGD step to update CPU matrix parameters
        lr = self.learning_rate
        for ent_id, grad in entity_grads.items():
            # Clip gradient to prevent exploding updates
            grad_clipped = np.clip(grad, -1.0, 1.0)
            self.entity_emb_matrix[ent_id] -= lr * grad_clipped

    def _prepare_data(self, X_body, X_body_names, X_head, X_head_names, anchor_labels):
        if os.path.exists(self.neo4j_env_path):
            load_dotenv(self.neo4j_env_path)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            load_dotenv(os.path.join(base_dir, '.env'))
            
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        pw = os.getenv('NEO4J_PASSWORD')
        
        body_paths = []
        head_paths = []
        
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
                    b_path = self._get_pattern_structure(X_body[i], X_body_names[i], anchor_labels[i], session)
                    h_path = self._get_pattern_structure(X_head[i], X_head_names[i], anchor_labels[i], session)
                    body_paths.append(b_path)
                    head_paths.append(h_path)
            driver.close()
        except Exception as e:
            print(f"\n[CRITICAL WARNING] Neo4j connection error: {e}")
            print("[CRITICAL WARNING] Falling back to zero embeddings for all patterns. This model will likely NOT learn correctly.")
            body_paths = [[] for _ in range(len(X_body))]
            head_paths = [[] for _ in range(len(X_body))]
        
        # Save cache if it has grown
        if len(self.pattern_cache) > initial_cache_size:
            self._save_persistent_cache()
            
        return body_paths, head_paths

    def fit(self, X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels, y, X_metrics=None, validation_split=0.2):
        print("Preprocessing patterns with Neo4j and RotatE embeddings...")
        body_paths_struct, head_paths_struct = self._prepare_data(X_body, X_body_names, X_head, X_head_names, anchor_labels)
        
        nentity = len(self.entity_dict)
        if nentity == 0 and self.entity_emb_matrix is not None:
            nentity = self.entity_emb_matrix.shape[0]
            print(f"[WARNING] entity_dict is empty, using nentity={nentity} from loaded checkpoint embedding matrix.")
        elif nentity == 0:
            nentity = 1000000

        rotate_params = {
            'nentity': nentity,
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
        
        # Setup validation structures
        val_body_paths = [body_paths_struct[j] for j in iv]
        val_head_paths = [head_paths_struct[j] for j in iv]
        
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
                
                # Fetch batch structures and generate embeddings dynamically
                batch_body_paths = [body_paths_struct[it[j]] for j in idx]
                batch_head_paths = [head_paths_struct[it[j]] for j in idx]
                
                batch_body_embs = self._compute_embeddings_from_structure(batch_body_paths)
                batch_head_embs = self._compute_embeddings_from_structure(batch_head_paths)
                
                # Detach and track gradients for input back-propagation
                body_t = batch_body_embs.to(self.device).detach().requires_grad_(True)
                head_t = batch_head_embs.to(self.device).detach().requires_grad_(True)
                
                optimizer.zero_grad()
                out = self.model(
                    [X_text[it[j]] for j in idx], 
                    body_t, 
                    head_t,
                    metrics=metrics_t[idx] if metrics_t is not None else None
                ).squeeze()
                if out.dim() == 0: out = out.unsqueeze(0)
                loss = criterion(out, y_t[idx])
                loss.backward()
                optimizer.step()
                
                # Distribute gradients back to CPU embeddings
                if body_t.grad is not None and head_t.grad is not None:
                    body_grad = body_t.grad.cpu().numpy()
                    head_grad = head_t.grad.cpu().numpy()
                    self._update_kge_embeddings(body_grad, head_grad, batch_body_paths, batch_head_paths)
                    
                total_train_loss += loss.item() * len(idx)
            
            avg_train_loss = total_train_loss / len(it)
            self.history_['train_loss'].append(avg_train_loss)
            
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                # Re-compute validation embeddings using updated CPU embeddings
                body_rotate_embs_v = self._compute_embeddings_from_structure(val_body_paths).to(self.device)
                head_rotate_embs_v = self._compute_embeddings_from_structure(val_head_paths).to(self.device)
                
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
        body_paths_struct, head_paths_struct = self._prepare_data(X_body, X_body_names, X_head, X_head_names, anchor_labels)
        
        # Dynamically compute embeddings using the latest fine-tuned CPU embedding matrix
        body_rotate_embs = self._compute_embeddings_from_structure(body_paths_struct).to(self.device)
        head_rotate_embs = self._compute_embeddings_from_structure(head_paths_struct).to(self.device)
        
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
