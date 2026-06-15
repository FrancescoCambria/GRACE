import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin

# Monkeypatch tensorflow.python.keras to avoid import errors in DeepCTR with newer TF versions
import tensorflow.keras as tf_keras
sys.modules['tensorflow.python.keras'] = tf_keras
sys.modules['tensorflow.python.keras.layers'] = tf_keras.layers
sys.modules['tensorflow.python.keras.models'] = tf_keras.models
sys.modules['tensorflow.python.keras.initializers'] = tf_keras.initializers
sys.modules['tensorflow.python.keras.regularizers'] = tf_keras.regularizers
sys.modules['tensorflow.python.keras.constraints'] = tf_keras.constraints
sys.modules['tensorflow.python.keras.utils'] = tf_keras.utils
sys.modules['tensorflow.python.keras.backend'] = tf_keras.backend

# Now we can safely import DeepCTR
from deepctr.models import WDL, DeepFM
from deepctr.feature_column import DenseFeat, get_feature_names

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class DeepCTRWideDeepClassifierWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags, ClassifierTags
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(self, dnn_hidden_units=(128, 128), epochs=100, batch_size=32, learning_rate=0.001):
        self.dnn_hidden_units = dnn_hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        feature_columns = [DenseFeat("feat", X.shape[1])]
        
        self.model = WDL(linear_feature_columns=feature_columns, 
                         dnn_feature_columns=feature_columns, 
                         dnn_hidden_units=self.dnn_hidden_units, 
                         task='binary')
        
        # Handle both eager and symbolic (TF1 legacy) modes
        if tf.executing_eagerly():
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            # Use legacy optimizer if eager execution is disabled (e.g. by another model)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            
        self.model.compile(optimizer, "binary_crossentropy", metrics=['accuracy'])
        
        train_model_input = {"feat": X}
        self.model.fit(train_model_input, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
        return self

    def predict(self, X):
        predict_model_input = {"feat": X}
        pred = self.model.predict(predict_model_input, batch_size=self.batch_size, verbose=0)
        return (pred > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        predict_model_input = {"feat": X}
        pred = self.model.predict(predict_model_input, batch_size=self.batch_size, verbose=0)
        # DeepCTR returns probability for class 1
        return np.hstack([1-pred, pred])

class DeepCTRDeepFMClassifierWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags, ClassifierTags
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(self, dnn_hidden_units=(128, 128), epochs=100, batch_size=32, learning_rate=0.001):
        self.dnn_hidden_units = dnn_hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        feature_columns = [DenseFeat("feat", X.shape[1])]
        
        self.model = DeepFM(linear_feature_columns=feature_columns, 
                           dnn_feature_columns=feature_columns, 
                           dnn_hidden_units=self.dnn_hidden_units, 
                           task='binary')
        
        # Handle both eager and symbolic (TF1 legacy) modes
        if tf.executing_eagerly():
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            
        self.model.compile(optimizer, "binary_crossentropy", metrics=['accuracy'])
        
        train_model_input = {"feat": X}
        self.model.fit(train_model_input, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
        return self

    def predict(self, X):
        predict_model_input = {"feat": X}
        pred = self.model.predict(predict_model_input, batch_size=self.batch_size, verbose=0)
        return (pred > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        predict_model_input = {"feat": X}
        pred = self.model.predict(predict_model_input, batch_size=self.batch_size, verbose=0)
        return np.hstack([1-pred, pred])
