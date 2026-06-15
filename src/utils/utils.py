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
