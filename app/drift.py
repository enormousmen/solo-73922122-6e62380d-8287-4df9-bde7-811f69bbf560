"""Distribution-drift detection via the Population Stability Index (PSI).

Baseline (training distribution)
--------------------------------
For every feature column of the CSV *actually loaded* by the service, five
quantile buckets are built from interior cut points at the 20/40/60/80%
quantiles of that column.  Coincident cuts (e.g. discrete or constant
columns) are merged, so a column can end up with fewer than five buckets;
the two end buckets are open-ended, so values outside the observed min/max
range naturally fall into them.

PSI per column
--------------
Both the baseline and the incoming counts are turned into proportions,
``1e-6`` is added to every bucket and the proportions are renormalized to sum
to 1.  The column score is then::

    PSI = sum_buckets (p_new - p_base) * ln(p_new / p_base)

A column with PSI > 0.2 raises an alert.

Baseline immutability / restart safety
--------------------------------------
:class:`DriftBaseline` is a frozen object containing only plain Python types,
so it travels verbatim inside the training checkpoint
(``torch.load(..., weights_only=True)`` can read it).  After a restart the
baseline is restored from the checkpoint and is *never* recomputed from the
CSV on disk — even if that file has been rewritten.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch

#: Target number of quantile buckets per column (interior cuts at 20/40/60/80%).
NUM_BUCKETS = 5
#: Additive smoothing before renormalization (applied to both distributions).
PSI_EPSILON = 1e-6
#: A column above this PSI is flagged as drifting.
PSI_ALERT_THRESHOLD = 0.2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_dataset_file(path: str) -> str:
    """SHA-256 of the raw CSV bytes, used to detect post-baseline rewrites."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_cuts(values: np.ndarray) -> list[float]:
    """Interior quantile cuts at 20/40/60/80%, with coincident cuts merged.

    The four cuts define five quantile buckets (``k`` cuts -> ``k+1``
    buckets).  When adjacent quantiles coincide (discrete / constant
    columns) the duplicates are dropped and the column gets fewer buckets;
    a constant column (all values equal) yields no cut and therefore a
    single catch-all bucket.  The observed min/max (the 0%/100% quantiles)
    are stored separately by :func:`build_drift_baseline`.
    """
    array = np.asarray(values, dtype=np.float64)
    interior_quantiles = [i / NUM_BUCKETS for i in range(1, NUM_BUCKETS)]
    cuts = np.quantile(array, interior_quantiles, method="linear")
    vmin = float(array.min())
    merged: list[float] = []
    for cut in cuts:
        cut = float(cut)
        # A cut sitting on the column minimum only happens when a large
        # share of values equals that minimum; it still separates buckets,
        # except for a fully constant column, where every cut equals both
        # the min and the max and must be dropped.
        if cut == vmin and cut == float(array.max()):
            continue
        if not merged or cut != merged[-1]:
            merged.append(cut)
    return merged


def num_buckets(cuts: list[float]) -> int:
    """Number of buckets defined by (possibly merged) quantile cuts.

    ``k`` distinct interior cuts define ``k+1`` buckets; a constant column
    with no cuts keeps one catch-all bucket.
    """
    return len(cuts) + 1


def assign_bucket(value: float, cuts: list[float]) -> int:
    """Map one value to a bucket index in ``0 .. num_buckets(cuts) - 1``.

    With cuts ``[c1, ..., ck]`` the buckets are
    ``(-inf, c1), [c1, c2), ..., [c_{k-1}, ck), [ck, +inf)``: a value equal
    to a cut goes to the bucket on its right, and values outside the
    observed training range naturally land in the two end buckets.  A
    cutless (constant) column maps every value to bucket 0.
    """
    for j, cut in enumerate(cuts):
        if value < cut:
            return j
    return len(cuts)


def bucket_counts(values: np.ndarray, cuts: list[float]) -> np.ndarray:
    """Count how many values fall into each bucket defined by ``cuts``."""
    counts = np.zeros(num_buckets(cuts), dtype=np.int64)
    for value in np.asarray(values, dtype=np.float64).tolist():
        counts[assign_bucket(float(value), cuts)] += 1
    return counts


def smoothed_proportions(counts: np.ndarray) -> np.ndarray:
    """Proportions with ``PSI_EPSILON`` added, renormalized to sum to 1."""
    smoothed = counts.astype(np.float64) + PSI_EPSILON
    return smoothed / smoothed.sum()


def psi_from_counts(base_counts: np.ndarray, new_counts: np.ndarray) -> float:
    """PSI between two count vectors over identical buckets."""
    p_base = smoothed_proportions(np.asarray(base_counts, dtype=np.int64))
    p_new = smoothed_proportions(np.asarray(new_counts, dtype=np.int64))
    return float(np.sum((p_new - p_base) * np.log(p_new / p_base)))


@dataclass(frozen=True)
class DriftBaseline:
    """Immutable per-run drift baseline, persisted inside the checkpoint."""

    run_id: str
    dataset_path: str
    dataset_sha256: str
    dataset_num_samples: int
    feature_columns: tuple[str, ...]
    #: Interior quantile cuts per feature (coincident cuts already merged).
    cuts: tuple[tuple[float, ...], ...]
    #: Observed training min/max per feature (0%/100% quantiles).
    observed_min: tuple[float, ...]
    observed_max: tuple[float, ...]
    #: Baseline bucket counts per feature (length len(cuts[j]) + 1).
    baseline_counts: tuple[tuple[int, ...], ...]
    created_at: str

    @property
    def num_features(self) -> int:
        return len(self.feature_columns)

    def bucket_count(self, feature_index: int) -> int:
        return num_buckets(list(self.cuts[feature_index]))

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "dataset_num_samples": self.dataset_num_samples,
            "feature_columns": list(self.feature_columns),
            "cuts": [list(c) for c in self.cuts],
            "observed_min": list(self.observed_min),
            "observed_max": list(self.observed_max),
            "baseline_counts": [list(c) for c in self.baseline_counts],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DriftBaseline":
        return cls(
            run_id=str(data["run_id"]),
            dataset_path=str(data["dataset_path"]),
            dataset_sha256=str(data["dataset_sha256"]),
            dataset_num_samples=int(data["dataset_num_samples"]),
            feature_columns=tuple(data["feature_columns"]),
            cuts=tuple(tuple(float(x) for x in c) for c in data["cuts"]),
            observed_min=tuple(float(x) for x in data["observed_min"]),
            observed_max=tuple(float(x) for x in data["observed_max"]),
            baseline_counts=tuple(
                tuple(int(x) for x in c) for c in data["baseline_counts"]),
            created_at=str(data.get("created_at", "")),
        )


def build_drift_baseline(
    X: torch.Tensor,
    feature_columns: list[str],
    dataset_path: str,
    run_id: str,
) -> DriftBaseline:
    """Build the immutable baseline from the currently loaded training CSV."""
    num_samples, num_features = int(X.shape[0]), int(X.shape[1])
    cuts_per_feature: list[tuple[float, ...]] = []
    mins: list[float] = []
    maxs: list[float] = []
    counts_per_feature: list[tuple[int, ...]] = []
    for j in range(num_features):
        column = X[:, j].numpy()
        cuts = quantile_cuts(column)
        counts = bucket_counts(column, cuts)
        cuts_per_feature.append(tuple(cuts))
        mins.append(float(column.min()))
        maxs.append(float(column.max()))
        counts_per_feature.append(tuple(int(c) for c in counts.tolist()))
    return DriftBaseline(
        run_id=run_id,
        dataset_path=dataset_path,
        dataset_sha256=hash_dataset_file(dataset_path),
        dataset_num_samples=num_samples,
        feature_columns=tuple(feature_columns),
        cuts=tuple(cuts_per_feature),
        observed_min=tuple(mins),
        observed_max=tuple(maxs),
        baseline_counts=tuple(counts_per_feature),
        created_at=_now_iso(),
    )


def score_drift(baseline: DriftBaseline,
                samples: list[list[float]]) -> dict:
    """Compute per-column PSI for one incoming batch against the baseline.

    ``samples`` rows must already be validated (finite, rectangular, width
    matching the baseline feature count).
    """
    if not samples:
        raise ValueError("samples must not be empty")
    width = len(samples[0])
    if width != baseline.num_features:
        raise ValueError(
            f"each sample must have {baseline.num_features} feature values "
            f"matching the training CSV columns {list(baseline.feature_columns)}, "
            f"got {width}")

    incoming = np.asarray(samples, dtype=np.float64)
    columns: list[dict] = []
    alerts: list[str] = []
    max_psi = 0.0
    for j, name in enumerate(baseline.feature_columns):
        cuts = list(baseline.cuts[j])
        base_counts = np.asarray(baseline.baseline_counts[j], dtype=np.int64)
        new_counts = bucket_counts(incoming[:, j], cuts)
        p_base = smoothed_proportions(base_counts)
        p_new = smoothed_proportions(new_counts)
        psi = psi_from_counts(base_counts, new_counts)
        is_alert = psi > PSI_ALERT_THRESHOLD
        if is_alert:
            alerts.append(name)
        max_psi = max(max_psi, psi)
        out_of_range = int(
            np.sum(incoming[:, j] < baseline.observed_min[j])
            + np.sum(incoming[:, j] > baseline.observed_max[j]))
        columns.append({
            "column": name,
            "bucket_count": num_buckets(cuts),
            "quantile_cuts": cuts,
            "observed_min": baseline.observed_min[j],
            "observed_max": baseline.observed_max[j],
            "baseline_counts": base_counts.tolist(),
            "incoming_counts": new_counts.tolist(),
            "baseline_proportions": p_base.tolist(),
            "incoming_proportions": p_new.tolist(),
            "out_of_range_count": out_of_range,
            "psi": psi,
            "alert": is_alert,
        })
    return {
        "num_samples": len(samples),
        "psi_threshold": PSI_ALERT_THRESHOLD,
        "columns": columns,
        "alerts": alerts,
        "max_psi": max_psi,
    }
