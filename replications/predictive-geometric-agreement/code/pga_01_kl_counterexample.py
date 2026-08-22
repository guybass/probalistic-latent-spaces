"""Independent numeric audit of the manuscript's regular KL counterexample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def bernoulli_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lower, upper = (float(value) for value in config["domain"])
    grid = np.linspace(lower, upper, int(config["grid_points"]), dtype=np.float64)
    base = 1.0 / 3.0 + grid / 20.0

    records = []
    minimum_probability = 1.0
    for raw_n in config["n_values"]:
        n = int(raw_n)
        candidate = base + np.sin(n * grid) / (100.0 * n)
        minimum_probability = min(
            minimum_probability,
            float(np.min(candidate)),
            float(np.min(1.0 - candidate)),
        )
        max_kl = float(np.max(bernoulli_kl(base, candidate)))
        records.append(
            {
                "n": n,
                "max_grid_kl": max_kl,
                "n_squared_max_grid_kl": n * n * max_kl,
            }
        )

    log_n = np.log([record["n"] for record in records])
    log_kl = np.log([record["max_grid_kl"] for record in records])
    kl_loglog_slope = float(np.polyfit(log_n, log_kl, 1)[0])

    r_zero = 1.0 / 3.0
    base_derivative = 1.0 / 20.0
    candidate_derivative = base_derivative + 1.0 / 100.0
    g_star = base_derivative**2 / (r_zero * (1.0 - r_zero))
    g_n = candidate_derivative**2 / (r_zero * (1.0 - r_zero))
    expected = config["expected"]
    errors = {
        "g_star_at_zero": abs(g_star - float(expected["g_star_at_zero"])),
        "g_n_at_zero": abs(g_n - float(expected["g_n_at_zero"])),
    }
    max_exact_metric_error = max(errors.values())

    acceptance = config["acceptance"]
    checks = {
        "exact_metric_constants": (
            max_exact_metric_error <= float(acceptance["max_exact_metric_error"])
        ),
        "uniform_interiority": (
            minimum_probability >= float(acceptance["minimum_probability_floor"])
        ),
        "quadratic_kl_rate": (
            float(acceptance["kl_loglog_slope_min"])
            <= kl_loglog_slope
            <= float(acceptance["kl_loglog_slope_max"])
        ),
        "metric_nonconvergence": abs(g_n - g_star) > 0.0,
    }
    payload = {
        "schema_version": 1,
        "target_id": "PGA-01",
        "method": {
            "base": "r(x)=1/3+x/20",
            "candidate": "r_n(x)=r(x)+sin(nx)/(100n)",
            "supremum_estimator": "fixed dense grid on the declared domain",
        },
        "results": {
            "g_star_at_zero": g_star,
            "g_n_at_zero": g_n,
            "metric_gap_at_zero": g_n - g_star,
            "max_exact_metric_error": max_exact_metric_error,
            "minimum_probability": minimum_probability,
            "kl_loglog_slope": kl_loglog_slope,
            "kl_grid_records": records,
        },
        "checks": checks,
        "accepted": all(checks.values()),
        "caveat": (
            "The grid supports the rate check but does not replace the paper's "
            "analytic uniform-interiority and Taylor argument."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["accepted"]:
        raise SystemExit("PGA-01 acceptance gate failed")


if __name__ == "__main__":
    main()
