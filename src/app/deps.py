import pickle
from pathlib import Path
from typing import Any, Optional

import yaml

from utils.logging import get_logger

log = get_logger("app.deps")


class ArtifactStore:
    def __init__(self):
        self.bm25 = None
        self.cfg = None


STORE = ArtifactStore()


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_artifacts(serve_config_path: str) -> None:
    cfg = load_config(serve_config_path)
    STORE.cfg = cfg

    bm25_path = Path(cfg["serve"]["bm25_artifact"])
    if bm25_path.exists():
        with bm25_path.open("rb") as f:
            STORE.bm25 = pickle.load(f)
        log.info("Loaded BM25 artifact: %s", bm25_path)
    else:
        STORE.bm25 = None
        log.info("BM25 artifact missing: %s (API will return empty hits)", bm25_path)

