"""
StreamLens — Calibrated LTR Scores
Converts raw LightGBM scores → calibrated relevance probabilities.

Why calibration matters:
  1. A/B testing: compare effect sizes in probability space
  2. Business dashboards: "item has 87% relevance probability"  
  3. Cost decisions: "only call GPT-4o-mini if relevance > 0.6"
  4. Threshold-based filtering: serve only p(relevance) > 0.5
"""
from __future__ import annotations
import pickle
import numpy as np

_PLATT   = None
_ISOTONIC = None

def _load():
    global _PLATT, _ISOTONIC
    if _PLATT is None:
        with open("artifacts/calibration/platt_scaler.pkl", "rb") as f:
            _PLATT = pickle.load(f)
        with open("artifacts/calibration/isotonic_scaler.pkl", "rb") as f:
            _ISOTONIC = pickle.load(f)


def calibrate(raw_scores: list[float], method: str = "platt") -> list[float]:
    """
    Convert raw LTR scores to calibrated relevance probabilities.
    
    Args:
        raw_scores: List of raw LightGBM LTR scores
        method:     "platt" (parametric) or "isotonic" (non-parametric)
    
    Returns:
        List of calibrated probabilities in [0, 1]
    
    Example:
        raw = [2.34, 1.87, 0.52, -0.31]
        cal = calibrate(raw)  # → [0.89, 0.73, 0.41, 0.18]
    """
    _load()
    arr = np.array(raw_scores).reshape(-1, 1)
    
    if method == "platt":
        return _PLATT.predict_proba(arr)[:, 1].tolist()
    else:
        return _ISOTONIC.predict(arr.flatten()).tolist()


def should_call_genai(calibrated_prob: float, threshold: float = 0.5) -> bool:
    """
    Cost guardrail: only call GPT-4o-mini if item is probably relevant.
    
    At threshold=0.5: skip GenAI on 50% of low-relevance items
    Saves ~$0.0004 per skipped explanation.
    """
    return calibrated_prob >= threshold
