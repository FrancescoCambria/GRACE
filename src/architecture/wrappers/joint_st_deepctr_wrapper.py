import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from deepctr_torch.models import WDL
from deepctr_torch.inputs import DenseFeat
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import copy
from sklearn.model_selection import train_test_split

class JointSTWDLModel(nn.Module):
    """
    Joint model combining SentenceTransformer and Wide & Deep (DeepCTR-Torch).
    This allows end-to-end training (fine-tuning) of both the embedding model and the classifier.
    """
    def __init__(self, st_model_name, use_st=True, use_metrics=False, metric_dim=16, dnn_hidden_units=(128, 128), device='cpu'):
        super().__init__()
        self.device = device
        self.use_st = use_st
        self.use_metrics = use_metrics
        
        total_dim = 0
        if use_st:
            self.st_model = SentenceTransformer(st_model_name).to(device)
            total_dim += self.st_model.get_sentence_embedding_dimension()
        
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
            
            total_dim += metric_dim
        
        # Define DenseFeat for DeepCTR-Torch
        self.feature_columns = [DenseFeat("feat", total_dim)]
        
        # WDL model from deepctr_torch
        self.wdl_model = WDL(linear_feature_columns=self.feature_columns, 
                             dnn_feature_columns=self.feature_columns, 
                             dnn_hidden_units=dnn_hidden_units, 
                             task='binary',
                             device=device)
        
    def forward(self, texts, metrics=None):
        parts = []
        
        if self.use_st:
            # 1. Tokenize texts
            features = self.st_model.tokenize(texts)
            features = {k: v.to(self.device) for k, v in features.items()}
            
            # 2. Extract embeddings (maintaining gradients)
            st_out = self.st_model(features)
            embeddings = st_out['sentence_embedding']
            parts.append(embeddings)
            
        if self.use_metrics and metrics is not None:
            support = metrics[:, 0].unsqueeze(1)
            confidence = metrics[:, 1].unsqueeze(1)
            supp_feats = self.support_proj(support)
            conf_feats = self.confidence_proj(confidence)
            metric_feats = torch.cat([supp_feats, conf_feats], dim=1)
            parts.append(metric_feats)
            
        if not parts:
            raise ValueError("No features available for forward pass. Enable use_st or use_metrics.")
            
        combined = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        
        # 3. Pass through WDL classifier
        return self.wdl_model(combined)

class JointSTDeepCTRWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for the Joint ST-WDL model.
    Accepts raw text strings as input (X).
    """
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags, ClassifierTags
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(self, st_model_name='all-MiniLM-L6-v2', use_st=True, use_metrics=False, metric_dim=16,
                 dnn_hidden_units=(128, 128), epochs=5, batch_size=16, 
                 learning_rate=1e-5, device=None, early_stopping_patience=10, 
                 use_lr_scheduler=False):
        self.st_model_name = st_model_name
        self.use_st = use_st
        self.use_metrics = use_metrics
        self.metric_dim = metric_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.use_lr_scheduler = use_lr_scheduler
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = None
        self.classes_ = None
        self.history_ = {'train_loss': [], 'val_loss': [], 'lr': []}

    def _to_list(self, X):
        if isinstance(X, np.ndarray):
            return X.tolist()
        return X

    def fit(self, X, y, X_metrics=None, validation_split=0.2):
        texts = self._to_list(X)
        self.classes_ = np.unique(y)
        
        # Split for validation
        indices = np.arange(len(texts))
        idx_train, idx_val = train_test_split(indices, test_size=validation_split, random_state=42, stratify=y)
        
        # Initialize model
        self.model = JointSTWDLModel(
            st_model_name=self.st_model_name, 
            use_st=self.use_st,
            use_metrics=self.use_metrics,
            metric_dim=self.metric_dim,
            dnn_hidden_units=self.dnn_hidden_units, 
            device=self.device
        )
        self.model.to(self.device)
        
        # Optimizer for both ST and WDL parameters
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Adaptive LR Scheduler
        scheduler = None
        if self.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        if X_metrics is not None:
            metrics_tensor = torch.tensor(X_metrics, dtype=torch.float32).to(self.device)
        else:
            metrics_tensor = None

        self.history_ = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            permutation = torch.randperm(len(idx_train))
            total_train_loss = 0
            for i in range(0, len(idx_train), self.batch_size):
                batch_idx = idx_train[permutation[i:i+self.batch_size]]
                batch_texts = [texts[idx] for idx in batch_idx]
                batch_y = y_tensor[batch_idx]
                batch_metrics = metrics_tensor[batch_idx] if metrics_tensor is not None else None
                
                optimizer.zero_grad()
                outputs = self.model(batch_texts, metrics=batch_metrics).squeeze()
                
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(batch_idx)
            
            avg_train_loss = total_train_loss / len(idx_train)
            self.history_['train_loss'].append(avg_train_loss)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for i in range(0, len(idx_val), self.batch_size):
                    batch_idx = idx_val[i:i+self.batch_size]
                    batch_texts = [texts[idx] for idx in batch_idx]
                    batch_y = y_tensor[batch_idx]
                    batch_metrics = metrics_tensor[batch_idx] if metrics_tensor is not None else None
                    
                    outputs = self.model(batch_texts, metrics=batch_metrics).squeeze()
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    
                    loss = criterion(outputs, batch_y)
                    total_val_loss += loss.item() * len(batch_idx)
            
            avg_val_loss = total_val_loss / len(idx_val)
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

    def predict(self, X, X_metrics=None):
        texts = self._to_list(X)
        self.model.eval()
        
        if X_metrics is not None:
            metrics_tensor = torch.tensor(X_metrics, dtype=torch.float32).to(self.device)
        else:
            metrics_tensor = None
            
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                batch_metrics = metrics_tensor[i:i+self.batch_size] if metrics_tensor is not None else None
                
                outputs = self.model(batch_texts, metrics=batch_metrics).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                preds = (outputs > 0.5).cpu().numpy().astype(int)
                all_preds.extend(preds)
        return np.array(all_preds)

    def predict_proba(self, X, X_metrics=None):
        texts = self._to_list(X)
        self.model.eval()
        
        if X_metrics is not None:
            metrics_tensor = torch.tensor(X_metrics, dtype=torch.float32).to(self.device)
        else:
            metrics_tensor = None
            
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                batch_metrics = metrics_tensor[i:i+self.batch_size] if metrics_tensor is not None else None
                
                outputs = self.model(batch_texts, metrics=batch_metrics).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                prob1 = outputs.cpu().numpy()
                all_probs.extend(prob1)
        
        prob1 = np.array(all_probs)
        # Ensure it's 2D for np.vstack
        if prob1.ndim == 0:
             prob1 = np.array([prob1])
        return np.vstack([1-prob1, prob1]).T
