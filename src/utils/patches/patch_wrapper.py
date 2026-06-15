import os

file_path = "core/wrappers/joint_rotate_wrapper.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Update JointSTRotatEModel __init__
content = content.replace(
    "def __init__(self, st_model_name, rotate_params, use_st=True, st_learned_dim=None, dnn_hidden_units=(128, 128), device='cpu'):",
    "def __init__(self, st_model_name, rotate_params, use_st=True, st_learned_dim=None, use_metrics=False, metric_dim=16, dnn_hidden_units=(128, 128), device='cpu'):"
)

content = content.replace(
    "self.st_out_dim = 0\n            \n        self.rotate_encoder = RotatEEncoder(**rotate_params).to(device)",
    "self.st_out_dim = 0\n            \n        self.rotate_encoder = RotatEEncoder(**rotate_params).to(device)\n        \n        self.use_metrics = use_metrics\n        self.metric_out_dim = metric_dim if use_metrics else 0\n        if use_metrics:\n            self.metric_proj = nn.Sequential(\n                nn.Linear(2, metric_dim),\n                nn.ReLU(),\n                nn.Linear(metric_dim, metric_dim)\n            ).to(device)"
)

content = content.replace(
    "total_dim = self.st_out_dim + 2 * self.rotate_encoder.out_dim\n        self.feature_columns = [DenseFeat(\"feat\", total_dim)]",
    "total_dim = self.st_out_dim + 2 * self.rotate_encoder.out_dim + self.metric_out_dim\n        self.feature_columns = [DenseFeat(\"feat\", total_dim)]"
)

# 2. Update JointSTRotatEModel forward
content = content.replace(
    "def forward(self, texts, body_rotate_embs, head_rotate_embs):",
    "def forward(self, texts, body_rotate_embs, head_rotate_embs, metrics=None):"
)

content = content.replace(
    "combined = torch.cat([st_emb, body_emb, head_emb], dim=1)\n        else:\n            combined = torch.cat([body_emb, head_emb], dim=1)\n            \n        return self.wdl_model(combined)",
    "parts = [st_emb, body_emb, head_emb]\n        else:\n            parts = [body_emb, head_emb]\n            \n        if self.use_metrics and metrics is not None:\n            parts.append(self.metric_proj(metrics))\n            \n        combined = torch.cat(parts, dim=1)\n        return self.wdl_model(combined)"
)

# 3. Update JointSTRotatEWrapper __init__
content = content.replace(
    "def __init__(self, use_st=True, st_model_name='all-MiniLM-L6-v2', st_learned_dim=None, rotate_hidden_dim=192, rotate_learned_dim=64, entities_dict_path=None, checkpoint_path=None, dnn_hidden_units=(128, 128), epochs=5, batch_size=16, learning_rate=1e-5, device=None, neo4j_env_path='/home/cambria/MineGraphRule/GRAM/.env', early_stopping_patience=10, use_lr_scheduler=False, use_instances=False, cache_path='kge/pattern_embeddings_cache.pkl'):",
    "def __init__(self, use_st=True, st_model_name='all-MiniLM-L6-v2', st_learned_dim=None, rotate_hidden_dim=192, rotate_learned_dim=64, use_metrics=False, entities_dict_path=None, checkpoint_path=None, dnn_hidden_units=(128, 128), epochs=5, batch_size=16, learning_rate=1e-5, device=None, neo4j_env_path='/home/cambria/MineGraphRule/GRAM/.env', early_stopping_patience=10, use_lr_scheduler=False, use_instances=False, cache_path='kge/pattern_embeddings_cache.pkl'):\n        self.use_metrics = use_metrics"
)

# 4. Update JointSTRotatEWrapper fit
content = content.replace(
    "def fit(self, X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels, y, validation_split=0.2):",
    "def fit(self, X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels, y, X_metrics=None, validation_split=0.2):"
)

content = content.replace(
    "st_learned_dim=self.st_learned_dim, \n            dnn_hidden_units=self.dnn_hidden_units,",
    "st_learned_dim=self.st_learned_dim, \n            use_metrics=getattr(self, 'use_metrics', False),\n            dnn_hidden_units=self.dnn_hidden_units,"
)

content = content.replace(
    "head_rotate_embs_v = head_rotate_embs[iv].to(self.device)\n        \n        self.history_ = {'train_loss': [], 'val_loss': [], 'lr': []}",
    "head_rotate_embs_v = head_rotate_embs[iv].to(self.device)\n        \n        if getattr(self, 'use_metrics', False) and X_metrics is not None:\n            metrics_t = torch.tensor(X_metrics[it], dtype=torch.float32).to(self.device)\n            metrics_v = torch.tensor(X_metrics[iv], dtype=torch.float32).to(self.device)\n        else:\n            metrics_t, metrics_v = None, None\n            \n        self.history_ = {'train_loss': [], 'val_loss': [], 'lr': []}"
)

content = content.replace(
    "[X_text[it[j]] for j in idx], \n                    body_rotate_embs_t[idx], \n                    head_rotate_embs_t[idx]\n                ).squeeze()",
    "[X_text[it[j]] for j in idx], \n                    body_rotate_embs_t[idx], \n                    head_rotate_embs_t[idx],\n                    metrics=metrics_t[idx] if metrics_t is not None else None\n                ).squeeze()"
)

content = content.replace(
    "[X_text[iv[j]] for j in range(i, end)], \n                        body_rotate_embs_v[i:end], \n                        head_rotate_embs_v[i:end]\n                    ).squeeze()",
    "[X_text[iv[j]] for j in range(i, end)], \n                        body_rotate_embs_v[i:end], \n                        head_rotate_embs_v[i:end],\n                        metrics=metrics_v[i:end] if metrics_v is not None else None\n                    ).squeeze()"
)

# 5. Update predict_proba
content = content.replace(
    "def predict_proba(self, X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels):",
    "def predict_proba(self, X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels, X_metrics=None):"
)

content = content.replace(
    "head_rotate_embs = head_rotate_embs.to(self.device)\n        \n        probs = []",
    "head_rotate_embs = head_rotate_embs.to(self.device)\n        \n        if getattr(self, 'use_metrics', False) and X_metrics is not None:\n            metrics_tensor = torch.tensor(X_metrics, dtype=torch.float32).to(self.device)\n        else:\n            metrics_tensor = None\n            \n        probs = []"
)

content = content.replace(
    "X_text[i:end], \n                    body_rotate_embs[i:end], \n                    head_rotate_embs[i:end]\n                ).squeeze()",
    "X_text[i:end], \n                    body_rotate_embs[i:end], \n                    head_rotate_embs[i:end],\n                    metrics=metrics_tensor[i:end] if metrics_tensor is not None else None\n                ).squeeze()"
)

# 6. Update predict
content = content.replace(
    "def predict(self, X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels):",
    "def predict(self, X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels, X_metrics=None):"
)
content = content.replace(
    "probs = self.predict_proba(X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels)",
    "probs = self.predict_proba(X_text, X_body, X_body_names, X_head, X_head_names, anchor_labels, X_metrics=X_metrics)"
)

with open(file_path, "w") as f:
    f.write(content)

print("Patch applied to joint_rotate_wrapper.py")
