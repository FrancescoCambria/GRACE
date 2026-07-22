import pandas as pd
import numpy as np
import json
import ast
import re

def parse_embedding(x):
    """
    Parses an embedding from a string or list/numpy array.
    Handles space-separated strings, comma-separated strings, and JSON-formatted strings.
    """
    if isinstance(x, str):
        # Remove any np.float32() or np.float64() wrappers
        x = re.sub(r'np\.float\d*\(', '', x).replace(')', '')
        x = x.strip('[]')
        try:
            if ',' in x:
                return np.array([float(val.strip()) for val in x.split(',') if val.strip()])
            return np.fromstring(x, sep=' ')
        except:
            try:
                return np.array(ast.literal_eval('[' + x + ']'))
            except:
                try:
                    return np.array(json.loads('[' + x + ']'))
                except:
                    return np.array([])
    elif isinstance(x, (list, np.ndarray)):
        return np.array(x)
    return x

def get_type(rule_str):
    """
    Returns the rule type by removing instance-specific IDs and attributes.
    """
    if not isinstance(rule_str, str):
        return ""
    return re.sub(r'\{.*?\}', '', rule_str).strip()

def calculate_rule_type_distribution(df):
    """
    Calculates the distribution of rule types in the dataset.
    """
    temp_df = df.copy()
    temp_df['body_type'] = temp_df['Body'].apply(get_type)
    temp_df['head_type'] = temp_df['Head'].apply(get_type)
    temp_df['combined_rule_type'] = temp_df['body_type'] + " -> " + temp_df['head_type']
    
    type_counts = temp_df['combined_rule_type'].value_counts()
    total_instances = len(temp_df)
    distribution = type_counts / total_instances
    
    return distribution, total_instances

def calculate_node_metrics(df):
    """
    Calculates average node participation and unique node count.
    """
    def get_ids(ids_str):
        if not isinstance(ids_str, str) or not ids_str:
            return []
        return [i.strip() for i in ids_str.split(',') if i.strip()]

    if len(df) == 0:
        return 0, 0

    body_nodes = df['Body Node IDs'].apply(get_ids) if 'Body Node IDs' in df.columns else pd.Series([[]]*len(df))
    head_nodes = df['Head Node IDs'].apply(get_ids) if 'Head Node IDs' in df.columns else pd.Series([[]]*len(df))
    
    all_rule_nodes = body_nodes + head_nodes
    total_node_instances = all_rule_nodes.apply(len).sum()
    all_nodes_flat = [node_id for rule_nodes in all_rule_nodes for node_id in rule_nodes]
    unique_nodes = set(all_nodes_flat)
    num_unique_nodes = len(unique_nodes)
    
    total_rule_instances = len(df)
    
    if num_unique_nodes == 0 or total_rule_instances == 0:
        avg_participation = 0
    else:
        avg_participation = total_node_instances / (num_unique_nodes * total_rule_instances)
        
    return avg_participation, num_unique_nodes

def balanced_train_test_split(*arrays, test_size=None, train_size=None, random_state=42, stratify=None):
    """
    Split arrays or matrices into random train and test subsets such that
    the training subset has exactly 50% class 0 and 50% class 1,
    based on the target labels 'stratify' (which must be provided).
    """
    if stratify is None:
        from sklearn.model_selection import train_test_split as sk_split
        return sk_split(*arrays, test_size=test_size, train_size=train_size, random_state=random_state)
        
    y = np.array(stratify)
    n_samples = len(y)
    
    # Calculate train size
    if train_size is not None:
        if isinstance(train_size, float):
            n_train = int(train_size * n_samples)
        else:
            n_train = train_size
    elif test_size is not None:
        if isinstance(test_size, float):
            n_train = int((1.0 - test_size) * n_samples)
        else:
            n_train = n_samples - test_size
    else:
        n_train = int(0.2 * n_samples)
        
    k = n_train // 2
    
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    
    k = min(k, len(idx_0), len(idx_1))
    
    if k == 0:
        from sklearn.model_selection import train_test_split as sk_split
        return sk_split(*arrays, test_size=test_size, train_size=train_size, random_state=random_state, stratify=stratify)
        
    rng = np.random.default_rng(random_state)
    
    train_idx_0 = rng.choice(idx_0, size=k, replace=False)
    train_idx_1 = rng.choice(idx_1, size=k, replace=False)
    
    test_idx_0 = np.setdiff1d(idx_0, train_idx_0)
    test_idx_1 = np.setdiff1d(idx_1, train_idx_1)
    
    train_idx = np.concatenate([train_idx_0, train_idx_1])
    test_idx = np.concatenate([test_idx_0, test_idx_1])
    
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    
    res = []
    for a in arrays:
        if isinstance(a, np.ndarray):
            res.append(a[train_idx])
            res.append(a[test_idx])
        elif hasattr(a, 'iloc'):
            res.append(a.iloc[train_idx])
            res.append(a.iloc[test_idx])
        elif isinstance(a, list):
            res.append([a[i] for i in train_idx])
            res.append([a[i] for i in test_idx])
        else:
            arr = np.array(a)
            res.append(arr[train_idx])
            res.append(arr[test_idx])
            
    return tuple(res)


def set_seed(seed=42):
    """
    Sets global random seeds for Python, NumPy, and PyTorch to ensure 100% reproducibility.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


