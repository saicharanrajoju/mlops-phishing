"""
Data-drift monitoring — the piece that closes the MLOps loop.

Compares a *reference* dataset (the training feature store) against a *current*
dataset (recent production traffic) and flags drift that should trigger
retraining. Two complementary signals per feature:

  * KS-test  - non-parametric test for a change in distribution (low p = drift).
  * PSI      - Population Stability Index; the industry-standard drift magnitude
               (>0.2 ~ significant shift).

Evidently AI is the production-grade alternative (richer reports/dashboards);
this keeps the dependency footprint light while demonstrating the same concept.

CLI:
    python -m phishsentinel.monitoring.drift_monitor --reference data/phisingData.csv \
        --current path/to/current.csv
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from phishsentinel.constant.training_pipeline import TARGET_COLUMN

KS_THRESHOLD = 0.05
PSI_THRESHOLD = 0.2


@dataclass
class FeatureDrift:
    feature: str
    ks_p_value: float
    psi: float
    drifted: bool


def population_stability_index(expected, actual, buckets: int = 10) -> float:
    """PSI of `actual` relative to `expected`, using quantile bins from expected."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
    if len(edges) < 3:
        return 0.0  # too few distinct values to bin meaningfully

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_perc = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a_perc = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    ks_threshold: float = KS_THRESHOLD,
    psi_threshold: float = PSI_THRESHOLD,
) -> dict:
    """Per-feature KS + PSI drift report with an overall dataset-drift verdict."""
    features = [c for c in reference_df.columns if c != TARGET_COLUMN and c in current_df.columns]
    results: list[FeatureDrift] = []

    for col in features:
        ref, cur = reference_df[col].dropna(), current_df[col].dropna()
        ks_p = float(ks_2samp(ref, cur).pvalue)
        psi = population_stability_index(ref, cur)
        drifted = ks_p < ks_threshold or psi > psi_threshold
        results.append(FeatureDrift(col, round(ks_p, 6), round(psi, 6), drifted))

    n_drifted = sum(r.drifted for r in results)
    share = n_drifted / len(results) if results else 0.0
    return {
        "n_features": len(results),
        "n_drifted": n_drifted,
        "drift_share": round(share, 4),
        # Flag the dataset as drifted when a meaningful fraction of features moved.
        "dataset_drift": share >= 0.3,
        "features": [asdict(r) for r in results],
    }


def save_reports(report: dict, out_dir: str = "monitoring_reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "drift_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    rows = "".join(
        f"<tr class='{'drift' if x['drifted'] else 'ok'}'>"
        f"<td>{x['feature']}</td><td>{x['ks_p_value']}</td>"
        f"<td>{x['psi']}</td><td>{'DRIFT' if x['drifted'] else 'ok'}</td></tr>"
        for x in report["features"]
    )
    verdict = "DATASET DRIFT DETECTED" if report["dataset_drift"] else "No significant drift"
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>MLOps (Phishing) Drift Report</title>
<style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left}}
.drift{{background:#ffe5e5}}.ok{{background:#eafaef}}h1{{margin-bottom:.2rem}}</style></head>
<body><h1>MLOps (Phishing) Drift Report</h1>
<p><b>{verdict}</b> — {report['n_drifted']}/{report['n_features']} features drifted
(share {report['drift_share']}).</p>
<table><tr><th>Feature</th><th>KS p-value</th><th>PSI</th><th>Status</th></tr>{rows}</table>
</body></html>"""
    html_path = os.path.join(out_dir, "drift_report.html")
    with open(html_path, "w") as f:
        f.write(html)
    return json_path


def main():
    parser = argparse.ArgumentParser(description="MLOps (Phishing) data-drift monitor")
    parser.add_argument("--reference", default=os.path.join("data", "phisingData.csv"))
    parser.add_argument(
        "--current",
        default=None,
        help="Current dataset CSV. Defaults to the reference (self-check = no drift).",
    )
    args = parser.parse_args()

    reference_df = pd.read_csv(args.reference)
    current_df = pd.read_csv(args.current) if args.current else reference_df.copy()

    report = generate_drift_report(reference_df, current_df)
    path = save_reports(report)
    verdict = "DATASET DRIFT DETECTED" if report["dataset_drift"] else "No significant drift"
    print(f"{verdict}: {report['n_drifted']}/{report['n_features']} features drifted.")
    print(f"Reports written under '{os.path.dirname(path)}/'.")


if __name__ == "__main__":
    main()
