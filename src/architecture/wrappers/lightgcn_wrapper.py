import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import sparse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
if hasattr(tf, 'compat'):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import recommenders_project.recommenders.models.lightgcn.lightgcn_utils as lightgcn_utils

class LightGCNClassifierWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        from sklearn.utils._tags import Tags, ClassifierTags
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(self, 
                 n_layers=3, 
                 epochs=100, 
                 batch_size=32, 
                 learning_rate=0.001,
                 hidden_units=(128,),
                 model_dir='models/tmp_lightgcn'):
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.model_dir = model_dir
        self.model = None
        self.classes_ = None

    def _build_adj_mat(self, n_rules, y):
        # Build bipartite graph: Rules <-> Tags
        n_tags = len(self.classes_)
        tag_map = {tag: i for i, tag in enumerate(self.classes_)}
        
        # Row indices (rules), Col indices (tags, shifted by n_rules)
        rows = np.arange(n_rules)
        cols = np.array([tag_map[val] for val in y]) + n_rules
        
        # Build the full symmetric adjacency matrix
        # [ 0  R ]
        # [ R' 0 ]
        # where R is rule-tag interactions
        
        adj_size = n_rules + n_tags
        R = sparse.coo_matrix((np.ones(len(y)), (rows, cols)), shape=(adj_size, adj_size))
        adj = R + R.T
        
        # Normalize: D^-1/2 * A * D^-1/2
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        
        # Convert to TF sparse tensor
        norm_adj = norm_adj.tocoo()
        indices = np.stack([norm_adj.row, norm_adj.col], axis=1)
        return tf.sparse.SparseTensor(indices, norm_adj.data.astype(np.float32), norm_adj.shape)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_rules = X.shape[0]
        n_features = X.shape[1]
        
        # Create sparse adj mat (bipartite)
        self.adj_mat = self._build_adj_mat(n_rules, y)
        
        self.model = lightgcn_utils.build_model(
            n_features=n_features,
            n_classes=n_classes,
            n_layers=self.n_layers,
            learning_rate=self.learning_rate,
            hidden_units=self.hidden_units
        )
        
        # For training, we need rule features and their labels.
        # The adj_mat is global for the training set.
        # Note: This simple implementation uses the full adj_mat in every call.
        
        # We wrap the training in a way that passes the adj_mat
        # Since we use Keras, we can pass it directly or via a generator.
        
        # Create tag embeddings (placeholder/zeros for tags since we only have features for rules)
        # In a real bipartite GCN, tags would also have features or learnable embeddings.
        # Here we pad X with zeros for the tag nodes.
        X_padded = np.vstack([X, np.zeros((n_classes, n_features))])
        
        # We need a custom training loop or a modified model to handle the sparse adj_mat correctly
        # for full-batch graph convolution.
        
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        
        @tf.function
        def train_step(features, labels, adj):
            with tf.GradientTape() as tape:
                predictions = self.model([features, adj])
                # We only want predictions for the rule nodes (first n_rules)
                rule_predictions = predictions[:n_rules]
                loss = loss_fn(labels, rule_predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss

        # Simple full-batch training for graph
        for epoch in range(self.epochs):
            loss = train_step(tf.constant(X_padded, dtype=tf.float32), tf.constant(y, dtype=tf.int32), self.adj_mat)
            if epoch % 10 == 0:
                try:
                    loss_val = loss.numpy()
                    print(f"Epoch {epoch}, Loss: {loss_val}")
                except:
                    # If symbolic, we can't easily print loss here without a session
                    pass
                
        return self

    def predict(self, X):
        # Inductive prediction for new rules is tricky with GCN.
        # A common way is to append new rules to the graph.
        # For simplicity here, we'll use the learned weights on the features.
        # But wait, LightGCN relies on the graph.
        
        # Alternative: The model we built can take [X, identity_sparse]
        # or we just take the rule embeddings part of the model.
        
        # Let's get the predictions
        # To handle test data, we'll treat them as isolated nodes for now or 
        # just use the feature-based part of the trained MLP.
        
        # Real GCN prediction usually involves the test nodes in the graph.
        # Since we are following sklearn API, X can be any new data.
        
        # For this wrapper, let's assume we want to evaluate on the same graph or 
        # provide a feature-only fallback.
        
        n_test = X.shape[0]
        # Pad with zeros for tags to keep shape consistent
        X_padded = np.vstack([X, np.zeros((len(self.classes_), X.shape[1]))])
        # Identity adj for test nodes (isolated)
        adj_test = tf.sparse.eye(X_padded.shape[0])
        
        # Use direct model call to handle SparseTensor without batching issues
        predictions = self.model([X_padded.astype(np.float32), adj_test], training=False)
        rule_predictions = predictions[:n_test]
        
        if tf.executing_eagerly():
            return np.argmax(rule_predictions, axis=1)
        else:
            # Use Keras session to maintain trained weights
            sess = tf.compat.v1.keras.backend.get_session()
            return sess.run(tf.argmax(rule_predictions, axis=1))

    def predict_proba(self, X):
        n_test = X.shape[0]
        X_padded = np.vstack([X, np.zeros((len(self.classes_), X.shape[1]))])
        adj_test = tf.sparse.eye(X_padded.shape[0])
        
        # Use direct model call to handle SparseTensor without batching issues
        predictions = self.model([X_padded.astype(np.float32), adj_test], training=False)
        rule_predictions = predictions[:n_test]
        
        if tf.executing_eagerly():
            return rule_predictions.numpy()
        else:
            sess = tf.compat.v1.keras.backend.get_session()
            return sess.run(rule_predictions)
