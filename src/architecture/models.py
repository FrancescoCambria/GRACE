import os
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_model_configs(recommenders_path='recommenders'):
    # Add recommenders to path
    repo_root = os.path.abspath(recommenders_path)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    model_configs = {
        'Logistic Regression': lambda: LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM': lambda: SVC(probability=True, random_state=42),
        'Neural Network (MLP)': lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

    # 1. Wide & Deep (Recommenders)
    try:
        from src.architecture.wrappers.wide_deep_wrapper import WideDeepClassifierWrapper
        model_configs['Wide & Deep'] = lambda: WideDeepClassifierWrapper(
            dnn_hidden_units=(128, 128), epochs=100, batch_size=16, optimizer='adam', learning_rate=0.001
        )
    except Exception as e:
        print(f"Warning: Could not import Wide & Deep wrapper: {e}")

    # 2. LightGCN (Project)
    try:
        from src.architecture.wrappers.lightgcn_wrapper import LightGCNClassifierWrapper
        model_configs['LightGCN (Project)'] = lambda: LightGCNClassifierWrapper(
            n_layers=3, epochs=100, learning_rate=0.001, hidden_units=(128, 128)
        )
    except Exception as e:
        print(f"Warning: Could not import LightGCN (Project) wrapper: {e}")

    # 3. LightGCN (Official Microsoft)
    try:
        from src.architecture.wrappers.lightgcn_official_wrapper import LightGCNOfficialClassifierWrapper
        model_configs['LightGCN (Official Microsoft)'] = lambda: LightGCNOfficialClassifierWrapper(
            n_layers=3, epochs=10, learning_rate=0.001
        )
    except Exception as e:
        print(f"Warning: Could not import LightGCN (Official Microsoft) wrapper: {e}")

    # 4. DeepCTR baseline wrappers (PyTorch-based WDL and DeepFM)
    try:
        from src.architecture.wrappers.deepctr_wrapper import DeepCTRWideDeepClassifierWrapper, DeepCTRDeepFMClassifierWrapper
        model_configs['DeepCTR Wide & Deep'] = lambda: DeepCTRWideDeepClassifierWrapper(
            dnn_hidden_units=(128, 128), epochs=100, batch_size=16, learning_rate=0.001
        )
        model_configs['DeepCTR DeepFM'] = lambda: DeepCTRDeepFMClassifierWrapper(
            dnn_hidden_units=(128, 128), epochs=100, batch_size=16, learning_rate=0.001
        )
    except Exception as e:
        print(f"Warning: Could not import DeepCTR baseline wrappers: {e}")

    # 5. Joint ST-DeepCTR
    try:
        from src.architecture.wrappers.joint_st_deepctr_wrapper import JointSTDeepCTRWrapper
        model_configs['Joint ST-DeepCTR'] = lambda: JointSTDeepCTRWrapper(
            epochs=5, batch_size=16, learning_rate=1e-5
        )
    except Exception as e:
        print(f"Warning: Could not import Joint ST-DeepCTR wrapper: {e}")

    # 6. Joint ST-Q2B-Avg
    try:
        from src.architecture.wrappers.joint_q2b_avg_st_wrapper import JointSTQ2BAvgDeepCTRWrapper
        model_configs['Joint ST-Q2B-Avg'] = lambda: JointSTQ2BAvgDeepCTRWrapper(
            epochs=5, batch_size=16, learning_rate=1e-5,
            entities_dict_path='kge/data/custom_dataset/entities.dict',
            relations_dict_path='kge/data/custom_dataset/relations.dict'
        )
    except Exception as e:
        print(f"Warning: Could not import Joint ST-Q2B-Avg wrapper: {e}")

    return model_configs
