import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
if hasattr(tf, 'compat'):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import recommenders_project.recommenders.models.wide_deep.wide_deep_utils as wide_deep_utils
from recommenders_project.recommenders.utils.tf_utils import pandas_input_fn, build_optimizer

class WideDeepClassifierWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    
    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags, ClassifierTags
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(self, 
                 dnn_hidden_units=(128, 128), 
                 epochs=100, 
                 batch_size=32, 
                 learning_rate=0.001,
                 optimizer='adam',
                 model_dir='models/tmp_wide_deep'):
        self.dnn_hidden_units = dnn_hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.model_dir = model_dir
        self.model = None
        self.classes_ = None

    def _prepare_df(self, X, y=None):
        # Create a DataFrame where the embedding is a single column of lists/arrays
        df = pd.DataFrame({'features': list(X)})
        if y is not None:
            df['label'] = y
        return df

    def fit(self, X, y):
        # Clean up old model directory
        if os.path.exists(self.model_dir):
            try:
                shutil.rmtree(self.model_dir)
            except:
                pass
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        df = self._prepare_df(X, y)
        
        # Use a single numeric column with the correct shape
        feature_columns = [
            tf.feature_column.numeric_column('features', shape=(X.shape[1],))
        ]
        
        self.model = wide_deep_utils.build_model(
            model_dir=self.model_dir,
            # For purely dense features, putting them in 'deep' is usually best
            # Wide part is typically for sparse features.
            wide_columns=[], 
            deep_columns=feature_columns,
            dnn_hidden_units=self.dnn_hidden_units,
            dnn_optimizer=build_optimizer(self.optimizer, lr=self.learning_rate),
            n_classes=n_classes,
            log_every_n_iter=100
        )
        
        self.model.train(
            input_fn=pandas_input_fn(
                df, 
                y_col='label', 
                batch_size=self.batch_size, 
                num_epochs=self.epochs,
                shuffle=True
            )
        )
        return self

    def predict(self, X):
        df = self._prepare_df(X)
        predictions = list(self.model.predict(
            input_fn=pandas_input_fn(df, batch_size=self.batch_size, num_epochs=1, shuffle=False)
        ))
        return np.array([p['class_ids'][0] for p in predictions])

    def predict_proba(self, X):
        df = self._prepare_df(X)
        predictions = list(self.model.predict(
            input_fn=pandas_input_fn(df, batch_size=self.batch_size, num_epochs=1, shuffle=False)
        ))
        return np.array([p['probabilities'] for p in predictions])
