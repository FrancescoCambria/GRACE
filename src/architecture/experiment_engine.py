import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer

def run_experiment(X_all, y_all, train_percent, dataset_name, model_name, model):
    train_size = train_percent / 100.0
    try:
        stratify = None
        unique_classes, counts = np.unique(y_all, return_counts=True)
        if len(unique_classes) > 1 and all(c >= 2 for c in counts):
             if train_size * len(y_all) >= len(unique_classes) and (1 - train_size) * len(y_all) >= len(unique_classes):
                stratify = y_all

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, train_size=train_size, random_state=42, stratify=stratify
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        try:
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = y_pred
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.5
            
        return {
            'Dataset': dataset_name,
            'Train %': train_percent,
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'AUC-ROC': auc
        }
    except Exception as e:
        return None

def run_cross_validation(X, y, model, name, n_splits=5):
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc'
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)
    
    return {
        'Model': name,
        'Accuracy': cv_results['test_accuracy'].mean(),
        'Precision': cv_results['test_precision'].mean(),
        'Recall': cv_results['test_recall'].mean(),
        'F1-Score': cv_results['test_f1'].mean(),
        'AUC-ROC': cv_results['test_roc_auc'].mean()
    }
