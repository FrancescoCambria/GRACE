# Classification for MineGraphRule

This project provides tools for analyzing, tagging, and classifying graph association rules extracted from various domains (currently Spotify and Law). It utilizes Natural Language Processing (NLP) and Knowledge Graph Embeddings (KGE) to represent rules and train classifiers.

## Project Structure (`src/`)

The codebase is organized into the `src/` directory to separate concerns and improve maintainability:

### `src/utils/`
Common utility modules and helper functions:
- `embedding_engine.py`: Engines for Gemini and SentenceTransformer rule embeddings.
- `translation.py`: Rule translation logic (MGR syntax to Natural Language).
- `utils.py`: General helper functions and embedding parsers.
- `memgraph_utils.py`: Utilities for interacting with Memgraph/Neo4j.
- `calculate_metrics.py`: Logic for dataset metric calculation.
- `visualize.py`: Plotting and result visualization logic.
- `maintenance/`: Maintenance scripts (e.g., `fix_rules_mismatch.py`).
- `patches/`: Temporary fixes for external libraries.

### `src/architecture/`
Core architectural components and model definitions:
- `models.py`: Model configurations and factory functions.
- `experiment_engine.py`: High-level logic for training and cross-validation.
- `kge_training.py`: Specific training logic for Knowledge Graph Embeddings.
- `get_pattern_embedding.py`: Utilities for extracting pattern embeddings.
- `train_schema_q2b.py`: Training logic for Q2B schema models.
- `wrappers/`: Classifier wrappers for various models (Joint Learning, RotatE, Q2B, DeepCTR, etc.).

### `src/experiments/`
Unified executable scripts for running experiments and data preparation:
- `analyze_and_tag.py`: Unified tool for analyzing criteria distributions and tagging datasets.
- `run_experiments.py`: Unified experiment runner (Individual runs, Grid Search, Joint & Baseline modes).
- `embed_rules.py`: Generates embeddings for rules in datasets.
- `extract_schema.py`: Extracts schema triples from various sources.
- `merge_rules.py`: Merges rule files from different extraction runs.
- `generate_noisy_datasets.py`: Injects noise into datasets for robustness testing.

## Getting Started

To run the scripts, ensure the project root is in your `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

1. **Tag your data**:
   ```bash
   python3 src/experiments/analyze_and_tag.py --input RulesSpotify/merged_variants/RulesSpotify_Merged.csv --criteria Basic4 --all-criteria
   ```

2. **Run Experiments**:
   ```bash
   # Single run with ST and RotatE
   python3 src/experiments/run_experiments.py --input RulesSpotify/LLMLogic/RulesSpotify_Basic4_1000.csv --include st rotate --lr 2e-5

   # Grid search over learning rates and test sizes
   python3 src/experiments/run_experiments.py --input RulesSpotify/LLMLogic/RulesSpotify_Basic4_1000.csv --include st metrics --lr 1e-5 2e-5 --test_size 0.2 0.5
   ```

Refer to `REORGANIZATION_LOG.md` for a detailed history of architectural changes.
