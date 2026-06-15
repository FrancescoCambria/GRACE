import os
import pandas as pd
import numpy as np
import argparse
import sys
import torch
import matplotlib.pyplot as plt
import json
import time
import itertools
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.utils.translation import translate_rule, translate_to_mgr_syntax
from src.architecture.wrappers.joint_st_deepctr_wrapper import JointSTDeepCTRWrapper
from src.architecture.wrappers.joint_q2b_st_wrapper import JointSTQ2BDeepCTRWrapper
from src.architecture.wrappers.joint_q2b_flexible_wrapper import JointSTQ2BFlexibleWrapper
from src.architecture.wrappers.joint_rotate_wrapper import JointSTRotatEWrapper
from src.utils.utils import parse_embedding
from src.architecture.models import get_model_configs
from src.architecture.experiment_engine import run_experiment, run_cross_validation

def plot_loss(history, save_path, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def check_data_leakage(df, idx_train, idx_test):
    def get_rule_identity(row):
        return f"{row['Body']}|{row['Body Node Names']} -> {row['Head']}|{row['Head Node Names']}"
    train_rules = set(df.iloc[idx_train].apply(get_rule_identity, axis=1))
    test_rules = set(df.iloc[idx_test].apply(get_rule_identity, axis=1))
    overlap = train_rules.intersection(test_rules)
    if overlap:
        print(f"[CRITICAL] Data Leakage: {len(overlap)} rules overlap!")
        return True
    return False

def run_joint_learning_task(args, dataset_path, run_id=1):
    """
    Logic ported from legacy run_joint_learning.py
    """
    df = pd.read_csv(dataset_path, sep=args.sep)
    essential_cols = ['Anchor Label', 'Body', 'Head', 'Body Node Names', 'Head Node Names']
    df = df.dropna(subset=essential_cols)
    
    # Deduplication to prevent leakage
    df = df.drop_duplicates(subset=['Body', 'Head', 'Body Node Names', 'Head Node Names'])
    
    if "mgr" in args.include:
        X_text = df.apply(lambda row: translate_to_mgr_syntax(row, f"Rule_{row.name}"), axis=1).values
    else:
        X_text = df.apply(translate_rule, axis=1).values
    
    y = df['tag'].values
    
    X_metrics = None
    if "metrics" in args.include:
        X_metrics_raw = df[['Support', 'Confidence']].fillna(0).values
        scaler = StandardScaler()
        X_metrics = scaler.fit_transform(X_metrics_raw)

    # Stratified split logic
    df['stratify_col'] = df['Body'].astype(str) + "|" + df['Head'].astype(str) + "|" + df['tag'].astype(str)
    counts = df['stratify_col'].value_counts()
    df['stratify_col'] = df.apply(lambda row: row['stratify_col'] if counts[row['stratify_col']] >= 2 else f"fallback_{row['tag']}", axis=1)
    
    indices = np.arange(len(X_text))
    split_random_state = 42 + run_id
    try:
        idx_train, idx_test = train_test_split(indices, test_size=args.test_size, random_state=split_random_state, stratify=df['stratify_col'].values)
    except:
        idx_train, idx_test = train_test_split(indices, test_size=args.test_size, random_state=split_random_state, stratify=y)

    if check_data_leakage(df, idx_train, idx_test):
        # We handle leakage by warning but continuing for now as per legacy behavior
        pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model selection
    use_st = "st" in args.include
    use_metrics = "metrics" in args.include
    use_rotate = "rotate" in args.include
    use_q2b = "q2b" in args.include

    model = None
    if use_rotate:
        model = JointSTRotatEWrapper(
            use_st=use_st, st_learned_dim=args.st_learned_dim, rotate_hidden_dim=args.kge_hidden_dim, 
            rotate_learned_dim=args.kge_learned_dim, use_metrics=use_metrics, metric_dim=args.metric_dim,
            entities_dict_path=args.entities_dict, checkpoint_path=args.checkpoint,
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device,
            early_stopping_patience=args.patience, use_lr_scheduler=args.use_lr_scheduler, use_instances=("instances" in args.include)
        )
        model.fit(X_text[idx_train], df['Body'].values[idx_train], df['Body Node Names'].values[idx_train],
                  df['Head'].values[idx_train], df['Head Node Names'].values[idx_train],
                  df['Anchor Label'].values[idx_train], y[idx_train],
                  X_metrics=X_metrics[idx_train] if X_metrics is not None else None)
        
        y_pred = model.predict(X_text[idx_test], df['Body'].values[idx_test], df['Body Node Names'].values[idx_test],
                               df['Head'].values[idx_test], df['Head Node Names'].values[idx_test],
                               df['Anchor Label'].values[idx_test], X_metrics=X_metrics[idx_test] if X_metrics is not None else None)
        y_prob = model.predict_proba(X_text[idx_test], df['Body'].values[idx_test], df['Body Node Names'].values[idx_test],
                                     df['Head'].values[idx_test], df['Head Node Names'].values[idx_test],
                                     df['Anchor Label'].values[idx_test], X_metrics=X_metrics[idx_test] if X_metrics is not None else None)[:, 1]

    elif use_q2b:
        model = JointSTQ2BFlexibleWrapper(
            mode=args.kge_mode, use_st=use_st, q2b_hidden_dim=args.kge_hidden_dim, q2b_learned_dim=args.kge_learned_dim,
            entities_dict_path=args.entities_dict, relations_dict_path=args.relations_dict, checkpoint_path=args.checkpoint,
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device,
            early_stopping_patience=args.patience, use_lr_scheduler=args.use_lr_scheduler
        )
        model.fit(X_text[idx_train], df['Body'].values[idx_train], df['Body Node Names'].values[idx_train],
                  df['Head'].values[idx_train], df['Head Node Names'].values[idx_train], y[idx_train])
        
        y_pred = model.predict(X_text[idx_test], df['Body'].values[idx_test], df['Body Node Names'].values[idx_test],
                               df['Head'].values[idx_test], df['Head Node Names'].values[idx_test])
        y_prob = model.predict_proba(X_text[idx_test], df['Body'].values[idx_test], df['Body Node Names'].values[idx_test],
                                     df['Head'].values[idx_test], df['Head Node Names'].values[idx_test])[:, 1]
    else:
        model = JointSTDeepCTRWrapper(
            use_st=use_st, use_metrics=use_metrics, metric_dim=args.metric_dim, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, device=device, early_stopping_patience=args.patience, use_lr_scheduler=args.use_lr_scheduler
        )
        model.fit(X_text[idx_train], y[idx_train], X_metrics=X_metrics[idx_train] if X_metrics is not None else None)
        y_pred = model.predict(X_text[idx_test], X_metrics=X_metrics[idx_test] if X_metrics is not None else None)
        y_prob = model.predict_proba(X_text[idx_test], X_metrics=X_metrics[idx_test] if X_metrics is not None else None)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y[idx_test], y_pred),
        'Precision': precision_score(y[idx_test], y_pred, zero_division=0),
        'Recall': recall_score(y[idx_test], y_pred, zero_division=0),
        'F1-Score': f1_score(y[idx_test], y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y[idx_test], y_prob)
    }
    
    actual_epochs = len(model.history_['train_loss']) if hasattr(model, 'history_') else 0
    return metrics, actual_epochs

def main():
    parser = argparse.ArgumentParser(description="Unified Experiment Runner")
    # Core Config
    parser.add_argument("--input", nargs="+", required=True, help="Input dataset path(s).")
    parser.add_argument("--sep", default=";", help="CSV separator.")
    parser.add_argument("--mode", choices=["joint", "baseline"], default="joint", help="Experiment mode.")
    parser.add_argument("--include", nargs="+", default=["st"], help="Components to include: st, rotate, q2b, metrics, instances, mgr.")
    
    # Hyperparameters (Supports multiple values for Grid Search)
    parser.add_argument("--lr", nargs="+", type=float, default=[2e-5], help="Learning rate(s).")
    parser.add_argument("--batch_size", nargs="+", type=int, default=[16], help="Batch size(s).")
    parser.add_argument("--test_size", nargs="+", type=float, default=[0.8], help="Test size(s).")
    parser.add_argument("--kge_learned_dim", nargs="+", type=int, default=[128], help="KGE learned dimension(s).")
    parser.add_argument("--st_learned_dim", type=int, default=None, help="ST learned dimension (None means no projection).")
    parser.add_argument("--metric_dim", type=int, default=16, help="Metrics learned dimension.")
    parser.add_argument("--patience", nargs="+", type=int, default=[20], help="Early stopping patience(s).")
    parser.add_argument("--include_configs", nargs="+", help="Comma-separated include configurations (e.g. 'st,metrics' 'st,rotate').")
    
    # Fixed Config
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per configuration.")
    parser.add_argument("--use_lr_scheduler", action="store_true", help="Enable LR scheduler.")
    
    # Paths & Dictionaries
    parser.add_argument("--checkpoint", help="KGE model checkpoint.")
    parser.add_argument("--entities_dict", default="kge/data/custom_dataset/entities.dict", help="Path to entities.dict")
    parser.add_argument("--relations_dict", default="kge/data/custom_dataset/relations.dict", help="Path to relations.dict")
    parser.add_argument("--kge_mode", default="average", help="Q2B mode (average, labelled, etc.)")
    parser.add_argument("--kge_hidden_dim", type=int, default=192, help="KGE hidden dimension.")
    
    # Output
    parser.add_argument("--output_report", default="reports/experiment_results.csv", help="CSV report path.")
    parser.add_argument("--log_file", default="reports/experiment_log.txt", help="Detailed log path.")

    args = parser.parse_args()

    # Generate Grid
    include_options = [args.include_configs] if args.include_configs else [",".join(args.include)]
    if args.include_configs:
        include_options = args.include_configs

    grid_params = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'test_size': args.test_size,
        'kge_learned_dim': args.kge_learned_dim,
        'patience': args.patience,
        'input': args.input,
        'include_str': include_options
    }
    
    keys, values = zip(*grid_params.items())
    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting {len(grid) * args.runs} experiments...")
    
    all_results = []
    
    for config in grid:
        # Update args with current grid config
        current_args = argparse.Namespace(**vars(args))
        for k, v in config.items():
            if k == 'include_str':
                current_args.include = v.split(",")
            else:
                setattr(current_args, k, v)
            
        print(f"\nConfig: {config} | Include: {current_args.include}")
        
        run_metrics = []
        start_time = time.time()
        
        for run_id in range(1, args.runs + 1):
            print(f"  Run {run_id}/{args.runs}...")
            try:
                if args.mode == "joint":
                    metrics, epochs = run_joint_learning_task(current_args, config['input'], run_id)
                else:
                    # Baseline logic
                    df = pd.read_csv(config['input'], sep=args.sep)
                    emb_col = 'Embedding' if 'Embedding' in df.columns else 'Combined_Embedding'
                    X_raw = df[emb_col].apply(parse_embedding).values
                    valid_indices = [i for i, emb in enumerate(X_raw) if emb.size > 0]
                    X = np.stack([X_raw[i] for i in valid_indices])
                    y = df['tag'].values[valid_indices]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    # For simplicity in unified runner, we just run Wide&Deep if in baseline mode
                    # or extend to loop through core.models if needed.
                    from src.architecture.wrappers.wide_deep_wrapper import WideDeepWrapper
                    res = run_experiment(X_scaled, y, int((1-config['test_size'])*100), os.path.basename(config['input']), "WideDeep", WideDeepWrapper())
                    metrics = {k: v for k, v in res.items() if k in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']}
                    epochs = 0
                
                metrics['epochs'] = epochs
                run_metrics.append(metrics)
            except Exception as e:
                print(f"    [ERROR] Run {run_id} failed: {e}")
                import traceback
                traceback.print_exc()

        if run_metrics:
            elapsed = (time.time() - start_time) / len(run_metrics)
            # Aggregate
            agg_row = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Dataset': os.path.basename(config['input']),
                'Include': "+".join(current_args.include),
                'LR': config['lr'],
                'BatchSize': config['batch_size'],
                'TestSize': config['test_size'],
                'Patience': config['patience'],
                'KG_Dim': config['kge_learned_dim'],
                'AvgTime': elapsed
            }
            
            for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
                vals = [rm[m] for rm in run_metrics]
                agg_row[f'Avg_{m}'] = np.mean(vals)
                agg_row[f'Max_{m}'] = np.max(vals)
            
            all_results.append(agg_row)
            
            # Log to file
            with open(args.log_file, "a") as f:
                f.write(f"{agg_row}\n")

    if all_results:
        os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
        report_df = pd.DataFrame(all_results)
        report_df.to_csv(args.output_report, index=False)
        print(f"\nDone! Report saved to {args.output_report}")

if __name__ == "__main__":
    main()
