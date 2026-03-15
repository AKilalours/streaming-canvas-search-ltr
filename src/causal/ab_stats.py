# src/causal/ab_stats.py
"""
A/B Statistical Testing with proper inference.
Computes t-tests, p-values, confidence intervals, and MDE.
This is what real experimentation platforms do before declaring a winner.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class ABTestResult:
    metric: str
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift_pct: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    mde: float          # minimum detectable effect at 80% power
    n_control: int
    n_treatment: int
    recommendation: str


def welch_t_test(
    control: list[float],
    treatment: list[float],
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Welch's t-test (unequal variances). Returns (t_stat, p_value)."""
    n1, n2 = len(control), len(treatment)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    m1 = sum(control) / n1
    m2 = sum(treatment) / n2
    v1 = sum((x - m1) ** 2 for x in control) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in treatment) / (n2 - 1)

    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0, 1.0

    t = (m2 - m1) / se

    # Welch-Satterthwaite degrees of freedom
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else 1.0

    # Approximate p-value using normal distribution for large df
    # For small df, this is a known approximation
    z = abs(t)
    if df > 30:
        p = 2 * (1 - _normal_cdf(z))
    else:
        p = 2 * _t_cdf_approx(z, df)

    return t, p


def _normal_cdf(z: float) -> float:
    """Approximation of standard normal CDF."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _t_cdf_approx(t: float, df: float) -> float:
    """Rough t-distribution tail probability approximation."""
    x = df / (df + t * t)
    # Incomplete beta function approximation
    return 0.5 * _regularized_incomplete_beta(df / 2, 0.5, x)


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Simple approximation via continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Use normal approximation for simplicity
    z = (x - a / (a + b)) / math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    return _normal_cdf(z)


def minimum_detectable_effect(
    n: int,
    baseline_mean: float,
    baseline_std: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Compute MDE: smallest lift detectable with given sample size and power.
    Uses normal approximation.
    """
    z_alpha = 1.96   # two-tailed alpha=0.05
    z_beta = 0.842   # power=0.80
    if n <= 0 or baseline_std <= 0:
        return float("inf")
    return (z_alpha + z_beta) * baseline_std * math.sqrt(2 / n)


def confidence_interval(
    values: list[float],
    alpha: float = 0.05,
) -> tuple[float, float]:
    """95% confidence interval for the mean."""
    n = len(values)
    if n < 2:
        return (0.0, 0.0)
    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    se = std / math.sqrt(n)
    z = 1.96  # 95% CI
    return (mean - z * se, mean + z * se)


def run_ab_test(
    control_rewards: list[float],
    treatment_rewards: list[float],
    metric_name: str = "avg_reward",
    alpha: float = 0.05,
) -> ABTestResult:
    """
    Full A/B test analysis with statistical inference.
    """
    n_c = len(control_rewards)
    n_t = len(treatment_rewards)

    if not control_rewards or not treatment_rewards:
        return ABTestResult(
            metric=metric_name, control_mean=0, treatment_mean=0,
            absolute_lift=0, relative_lift_pct=0, t_statistic=0,
            p_value=1.0, ci_lower=0, ci_upper=0, significant=False,
            mde=0, n_control=n_c, n_treatment=n_t,
            recommendation="Insufficient data",
        )

    ctrl_mean = sum(control_rewards) / n_c
    trt_mean = sum(treatment_rewards) / n_t
    abs_lift = trt_mean - ctrl_mean
    rel_lift = (abs_lift / abs(ctrl_mean) * 100) if ctrl_mean != 0 else 0.0

    t_stat, p_value = welch_t_test(control_rewards, treatment_rewards, alpha)
    ci_lo, ci_hi = confidence_interval(treatment_rewards)

    ctrl_std = math.sqrt(
        sum((x - ctrl_mean) ** 2 for x in control_rewards) / max(n_c - 1, 1)
    )
    mde = minimum_detectable_effect(n_c, ctrl_mean, ctrl_std)

    significant = p_value < alpha

    if significant and abs_lift > 0:
        rec = f"Ship treatment — statistically significant lift of {rel_lift:.1f}% (p={p_value:.4f})"
    elif significant and abs_lift < 0:
        rec = f"Do NOT ship — statistically significant regression of {rel_lift:.1f}% (p={p_value:.4f})"
    elif not significant and abs(abs_lift) < mde:
        rec = f"Underpowered — lift ({abs_lift:.4f}) below MDE ({mde:.4f}). Need more data."
    else:
        rec = f"Not significant (p={p_value:.4f}). Collect more data or accept null hypothesis."

    return ABTestResult(
        metric=metric_name,
        control_mean=round(ctrl_mean, 6),
        treatment_mean=round(trt_mean, 6),
        absolute_lift=round(abs_lift, 6),
        relative_lift_pct=round(rel_lift, 2),
        t_statistic=round(t_stat, 4),
        p_value=round(p_value, 6),
        ci_lower=round(ci_lo, 6),
        ci_upper=round(ci_hi, 6),
        significant=significant,
        mde=round(mde, 6),
        n_control=n_c,
        n_treatment=n_t,
        recommendation=rec,
    )
