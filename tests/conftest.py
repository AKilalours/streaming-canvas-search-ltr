import sys
from pathlib import Path

# Ensure "src/" is importable when running pytest directly (without PYTHONPATH=src)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
