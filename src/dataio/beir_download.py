from pathlib import Path
from typing import Tuple

from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip

from utils.io import ensure_dir
from utils.logging import get_logger

log = get_logger("dataio.beir_download")


def download_beir_dataset(dataset: str, raw_root: str) -> Path:
    raw_root_p = ensure_dir(raw_root)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = raw_root_p / dataset
    if out_dir.exists():
        log.info("BEIR dataset already present: %s", out_dir)
        return out_dir
    log.info("Downloading BEIR dataset: %s", dataset)
    zip_path = download_and_unzip(url, str(raw_root_p))
    # download_and_unzip returns path; dataset folder is raw_root/dataset
    log.info("Downloaded/unzipped: %s", zip_path)
    return out_dir


def load_beir_split(dataset_dir: Path, split: str):
    # split in {"train","dev","test"}
    corpus, queries, qrels = GenericDataLoader(str(dataset_dir)).load(split=split)
    return corpus, queries, qrels

