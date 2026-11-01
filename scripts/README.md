# Scripts

Run commands from the repository root unless a command says otherwise. Use the project environment (`uv run ...`) for Python scripts where practical.

## Maintained tooling

- `build/`: corpus, embedding, and MovieLens dataset/qrels preparation.
- `evaluation/`: offline metrics, latency/load/scale checks, drift monitoring, and shadow evaluation.
- `training/`: MovieLens learning-to-rank training entry point.
- `operations/`: Metaflow and local MinIO operational helpers.
- `testing/`: language and integration checks.
- `opensearch/`: OpenSearch indexing and hybrid retrieval evaluation.

## Experiments

Exploratory or one-off scripts live under `experiments/`, grouped by modality or purpose. They are not production entry points and may require optional services, datasets, or credentials. Read the script before running it; some experiments provision or alter external resources.

## Paths

Flow, Makefile, and documented commands should use the categorized paths above. Add new maintained scripts to the relevant category and update this index when adding a category.
