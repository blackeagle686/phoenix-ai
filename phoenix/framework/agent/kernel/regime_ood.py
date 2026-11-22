"""PHX v2 Regime Classifier & Out-of-Distribution (OOD) Guard (FR-02, Section 5.3).

Uses MiniBatch K-Means centroids + regularized Mahalanobis (or scaled Euclidean)
distance to classify noisy telemetry into clean operational regimes and prevent
Random Forest overconfidence on unseen regions (< 0.2ms CPU).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:  # pragma: no cover
    MiniBatchKMeans = None


class RegimeOODGuard:
    """Discretizes state vectors into regimes and guards against OOD inputs."""

    def __init__(self, n_clusters: int = 4, ood_percentile: float = 97.5, cov_reg: float = 1e-2):
        self.n_clusters = max(1, int(n_clusters))
        self.ood_percentile = float(ood_percentile)
        self.cov_reg = float(cov_reg)
        self.fitted: bool = False
        self.centroids: Optional[np.ndarray] = None          # (K, D)
        self.inv_covs: List[Optional[np.ndarray]] = []       # list of (D, D) or None
        self.scale_stds: Optional[np.ndarray] = None         # (D,)
        self.theta_ood: float = float("inf")
        self.cluster_counts: List[int] = []

    def fit(self, X: np.ndarray) -> "RegimeOODGuard":
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            self.fitted = False
            return self

        n_samples, dim = arr.shape
        k = min(self.n_clusters, max(1, n_samples // 3))
        if k <= 1:
            centroids = arr.mean(axis=0, keepdims=True)
            labels = np.zeros(n_samples, dtype=int)
        elif MiniBatchKMeans is not None:
            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=42,
                batch_size=min(256, n_samples),
                n_init=3,
            )
            labels = km.fit_predict(arr)
            centroids = km.cluster_centers_
        else:
            # Pure-NumPy K-Means fallback
            rng = np.random.default_rng(42)
            idx = rng.choice(n_samples, size=k, replace=False)
            centroids = arr[idx].copy()
            labels = np.zeros(n_samples, dtype=int)
            for _ in range(15):
                dists = np.linalg.norm(arr[:, None, :] - centroids[None, :, :], axis=2)
                labels = np.argmin(dists, axis=1)
                for ci in range(k):
                    pts = arr[labels == ci]
                    if len(pts) > 0:
                        centroids[ci] = pts.mean(axis=0)

        self.centroids = centroids.astype(np.float64)
        global_std = np.std(arr, axis=0)
        # Floor scale_stds at 0.20 so near-constant normalized [0, 1] telemetry scalars don't amplify tiny jitter into false OOD alarms
        self.scale_stds = np.maximum(global_std, 0.20)
        self.inv_covs = []
        self.cluster_counts = []

        for ci in range(self.centroids.shape[0]):
            pts = arr[labels == ci]
            self.cluster_counts.append(int(len(pts)))
            if len(pts) > dim + 2:
                diff = pts - self.centroids[ci]
                cov = (diff.T @ diff) / max(1, len(pts) - 1)
                cov += np.eye(dim) * self.cov_reg
                try:
                    cond = float(np.linalg.cond(cov))
                    if np.isfinite(cond) and cond < 1e6:
                        self.inv_covs.append(np.linalg.inv(cov))
                    else:
                        self.inv_covs.append(None)
                except np.linalg.LinAlgError:
                    self.inv_covs.append(None)
            else:
                self.inv_covs.append(None)

        # Calibrate OOD threshold from in-distribution training distances with dimensionality-aware floor
        train_dists = np.array([self._distance_to_cluster(arr[i], int(labels[i])) for i in range(n_samples)], dtype=np.float64)
        base_thresh = float(np.percentile(train_dists, self.ood_percentile))
        mad = float(np.median(np.abs(train_dists - np.median(train_dists))))
        dim_floor = max(2.2, 0.35 * float(np.sqrt(dim)))
        self.theta_ood = max(base_thresh * 1.25 + 3.0 * mad, dim_floor)
        self.fitted = True
        return self

    def _distance_to_cluster(self, x: np.ndarray, cluster_idx: int) -> float:
        assert self.centroids is not None and self.scale_stds is not None
        diff = x - self.centroids[cluster_idx]
        inv_cov = self.inv_covs[cluster_idx] if cluster_idx < len(self.inv_covs) else None
        if inv_cov is not None:
            val = float(diff @ inv_cov @ diff)
            return float(np.sqrt(max(0.0, val)))
        scaled = diff / self.scale_stds
        return float(np.linalg.norm(scaled))

    def evaluate(self, x_vec: np.ndarray) -> Tuple[int, float, bool]:
        """Return `(regime_id, ood_distance, is_ood)` in < 0.2ms CPU."""
        if not self.fitted or self.centroids is None:
            return 0, 0.0, False
        x = np.asarray(x_vec, dtype=np.float64).ravel()
        if x.size != self.centroids.shape[1]:
            return 0, float("inf"), True

        best_k = 0
        best_d = float("inf")
        for k in range(self.centroids.shape[0]):
            d = self._distance_to_cluster(x, k)
            if d < best_d:
                best_d = d
                best_k = k
        is_ood = bool(best_d > self.theta_ood)
        return best_k, round(best_d, 4), is_ood
