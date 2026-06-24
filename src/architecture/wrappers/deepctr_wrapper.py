import torch
import torch.nn as nn
import torch.optim as optim
from deepctr_torch.models import WDL, DeepFM
from deepctr_torch.inputs import DenseFeat
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class DeepCTRWideDeepClassifierWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags, ClassifierTags
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(self, dnn_hidden_units=(128, 128), epochs=100, batch_size=16, learning_rate=0.001, device=None):
        self.dnn_hidden_units = dnn_hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        feature_columns = [DenseFeat("feat", X.shape[1])]
        
        self.model = WDL(linear_feature_columns=feature_columns, 
                         dnn_feature_columns=feature_columns, 
                         dnn_hidden_units=self.dnn_hidden_units, 
                         task='binary',
                         device=self.device)
        self.model.to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        self.model.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(X.shape[0])
            for i in range(0, X.shape[0], self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_x = X_tensor[indices]
                batch_y = y_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        all_preds = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                batch_x = X_tensor[i:i+self.batch_size]
                outputs = self.model(batch_x).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                preds = (outputs > 0.5).cpu().numpy().astype(int)
                all_preds.extend(preds)
        return np.array(all_preds)

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        all_probs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                batch_x = X_tensor[i:i+self.batch_size]
                outputs = self.model(batch_x).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                probs = outputs.cpu().numpy()
                all_probs.extend(probs)
        
        prob1 = np.array(all_probs)
        if prob1.ndim == 0:
            prob1 = np.array([prob1])
        return np.vstack([1-prob1, prob1]).T

class DeepCTRDeepFMClassifierWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags, ClassifierTags
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(self, dnn_hidden_units=(128, 128), epochs=100, batch_size=16, learning_rate=0.001, device=None):
        self.dnn_hidden_units = dnn_hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        feature_columns = [DenseFeat("feat", X.shape[1])]
        
        self.model = DeepFM(linear_feature_columns=feature_columns, 
                            dnn_feature_columns=feature_columns, 
                            dnn_hidden_units=self.dnn_hidden_units, 
                            task='binary',
                            device=self.device)
        self.model.to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        self.model.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(X.shape[0])
            for i in range(0, X.shape[0], self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_x = X_tensor[indices]
                batch_y = y_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        all_preds = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                batch_x = X_tensor[i:i+self.batch_size]
                outputs = self.model(batch_x).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                preds = (outputs > 0.5).cpu().numpy().astype(int)
                all_preds.extend(preds)
        return np.array(all_preds)

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        all_probs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                batch_x = X_tensor[i:i+self.batch_size]
                outputs = self.model(batch_x).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                probs = outputs.cpu().numpy()
                all_probs.extend(probs)
        
        prob1 = np.array(all_probs)
        if prob1.ndim == 0:
            prob1 = np.array([prob1])
        return np.vstack([1-prob1, prob1]).T
