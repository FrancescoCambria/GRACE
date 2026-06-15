# Codebase Reorganization Log

The codebase has been reorganized into a more scalable and logical structure, following the user's directive to consolidate scripts and separate concerns.

## Final Structure (`src/`)

All core logic and executable scripts now reside within the `src/` directory, subdivided into three main areas:

### 1. `src/utils/`
Common utility modules and helper functions:
- `calculate_metrics.py`: Logic for dataset metric calculation.
- `embedding_engine.py`: Engines for Gemini and SentenceTransformer rule embeddings.
- `memgraph_utils.py`: Utilities for interacting with Memgraph/Neo4j.
- `translation.py`: Rule translation logic (MGR syntax to Natural Language).
- `utils.py`: General helper functions and embedding parsers.
- `visualize.py`: Plotting and result visualization logic.
- `maintenance/`: Maintenance scripts (e.g., `fix_rules_mismatch.py`).
- `patches/`: Temporary fixes and monkey-patches for external libraries.

### 2. `src/architecture/`
Core architectural components and model definitions:
- `models.py`: Model configurations and factory functions.
- `experiment_engine.py`: High-level logic for training and cross-validation.
- `kge_training.py`: Specific training logic for Knowledge Graph Embeddings.
- `get_pattern_embedding.py`: Utilities for extracting pattern embeddings.
- `train_schema_q2b.py`: Training logic for Q2B schema models.
- `wrappers/`: Classifier wrappers for various recommendation models:
  - `joint_rotate_wrapper.py`: Joint Learning with RotatE.
  - `joint_q2b_st_wrapper.py`: Joint Learning with Q2B and SentenceTransformers.
  - `joint_st_deepctr_wrapper.py`: Joint Learning with DeepCTR.
  - `lightgcn_wrapper.py`: LightGCN integration.
  - `wide_deep_wrapper.py`: Wide & Deep model integration.
  - (and others...)

### 3. `src/experiments/`
Unified executable scripts for running experiments and data preparation:
- `analyze_and_tag.py`: **Unified tool** for analyzing criteria distributions and tagging datasets. Replaces multiple legacy `retag` scripts.
- `run_experiments.py`: **Unified experiment runner**. Supports individual runs, grid search, joint learning, and baseline modes. Replaces `run_all_experiments.py` and legacy `run_joint_learning.py`.
- `embed_rules.py`: Generates embeddings for rules in datasets.
- `extract_schema.py`: Extracts schema triples from various sources.
- `merge_rules.py`: Merges rule files from different extraction runs.
- `generate_noisy_datasets.py`: Injects noise into datasets for robustness testing.

## Key Changes

- **Consolidated Scripting**: Reduced the number of top-level scripts by merging similar functionality into unified tools with flexible CLI arguments.
- **Improved Import System**: Scripts now use the `src.` package prefix. To run them, ensure the project root is in your `PYTHONPATH`.
- **Clean Workspace**: Removed redundant copies of scripts and organized all source code into a single `src/` hierarchy.
- **Backup**: A full backup of the previous state is maintained in `ClassificationforMineGraphRule_backup/`.

## Documentation Update (June 15, 2026)

- **README Synchronization**: The `README.md` has been fully updated to align with the new `src/` structure.
- **Usage Instructions**: Updated execution commands to use the new paths (e.g., `python3 src/experiments/run_experiments.py`) and added instructions for setting `PYTHONPATH`.
- **Structural Accuracy**: All component descriptions now match their new locations within `src/utils/`, `src/architecture/`, and `src/experiments/`.
