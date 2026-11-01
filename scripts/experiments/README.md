# Experimental scripts

These standalone scripts are organized by the area they explore. The maintained
application and reusable production components live in `src/`; these scripts are
for training, evaluation, and demo workflows that need local datasets, optional
model dependencies, credentials, or generated artifacts.

Run commands from the repository root so paths such as `data/` and `artifacts/`
resolve as written.

- `audio/`: local speech recognition experiments.
- `media/`: diffusion and poster generation experiments.
- `ranking/`: ranking, calibration, and clustering experiments.
- `recommendation/`: collaborative filtering and bandit experiments.
- `retrieval/`: query understanding, evaluation, and fine-tuning experiments.

These scripts are not imported by the API. Check a script's prerequisites and
side effects before running it; some train models or overwrite generated artifacts.
