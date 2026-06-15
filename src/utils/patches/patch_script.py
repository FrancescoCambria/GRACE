import os

file_path = "scripts/run_joint_learning.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Add --use_metrics
content = content.replace(
    'parser.add_argument("--test_size", type=float, default=0.8, help="Test size split ratio.")',
    'parser.add_argument("--test_size", type=float, default=0.8, help="Test size split ratio.")\n    parser.add_argument("--use_metrics", action="store_true", help="Include Support and Confidence as numerical features in the embedding.")'
)

# 2. Extract metrics
content = content.replace(
    "X_anchor_labels = df['Anchor Label'].values",
    "X_anchor_labels = df['Anchor Label'].values\n    \n    if args.use_metrics:\n        from sklearn.preprocessing import StandardScaler\n        X_metrics_raw = df[['Support', 'Confidence']].fillna(0).values\n        scaler = StandardScaler()\n        X_metrics = scaler.fit_transform(X_metrics_raw)\n    else:\n        X_metrics = None"
)

# 3. Update JointSTRotatEWrapper initialization
content = content.replace(
    "rotate_learned_dim=args.rotate_learned_dim,",
    "rotate_learned_dim=args.rotate_learned_dim,\n            use_metrics=args.use_metrics,"
)

# 4. Update fit call
content = content.replace(
    "X_anchor_labels[idx_train],\n            y[idx_train]\n        )",
    "X_anchor_labels[idx_train],\n            y[idx_train],\n            X_metrics=X_metrics[idx_train] if X_metrics is not None else None\n        )"
)

# 5. Update predict call
content = content.replace(
    "X_anchor_labels[idx_test]\n        )",
    "X_anchor_labels[idx_test],\n            X_metrics=X_metrics[idx_test] if X_metrics is not None else None\n        )"
)

# 6. Update predict_proba call
content = content.replace(
    "X_anchor_labels[idx_test]\n        )[:, 1]",
    "X_anchor_labels[idx_test],\n            X_metrics=X_metrics[idx_test] if X_metrics is not None else None\n        )[:, 1]"
)

with open(file_path, "w") as f:
    f.write(content)
print("Patch applied to run_joint_learning.py")
