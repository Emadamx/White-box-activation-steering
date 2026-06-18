"""
DirectionFinder: Learns a unit steering direction v̂ from contrastive
activation pairs (cooperative H^c vs. deceptive H^d).

Three methods are supported:
  - 'lp'  : Linear Probe (recommended — gives probe accuracy as a layer selection metric)
  - 'dom' : Difference of Means (fast baseline)
  - 'pca' : Top PCA component of the mean-centred activation difference
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class DirectionMethod(str, Enum):
    LINEAR_PROBE = "lp"
    DIFF_OF_MEANS = "dom"
    PCA = "pca"


class DirectionFinder:
    """
    Learns a unit direction v̂ ∈ ℝ^d that separates cooperative from
    deceptive activations in a single transformer layer.

    Args:
        method: One of 'lp', 'dom', 'pca'.
        normalize: Whether to L2-normalise activations before fitting.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        method: Literal["lp", "dom", "pca"] = "lp",
        normalize: bool = True,
        random_state: int = 42,
    ) -> None:
        self.method = DirectionMethod(method)
        self.normalize = normalize
        self.random_state = random_state

        self.direction_: Optional[np.ndarray] = None   # unit vector v̂
        self.probe_accuracy_: Optional[float] = None   # only set for 'lp'
        self._scaler: Optional[StandardScaler] = None

    def fit(
        self,
        H_cooperative: torch.Tensor,
        H_deceptive: torch.Tensor,
    ) -> "DirectionFinder":
        """
        Learn the steering direction from contrastive activation pairs.

        Args:
            H_cooperative: Activations from cooperative (monitored) rollouts.
                           Shape: (n_samples, d_model)
            H_deceptive:   Activations from deceptive (unmonitored) rollouts.
                           Shape: (n_samples, d_model)

        Returns:
            self (for chaining)
        """
        H_c = H_cooperative.numpy() if isinstance(H_cooperative, torch.Tensor) else H_cooperative
        H_d = H_deceptive.numpy() if isinstance(H_deceptive, torch.Tensor) else H_deceptive

        if self.normalize:
            self._scaler = StandardScaler()
            H_all = np.vstack([H_c, H_d])
            H_all = self._scaler.fit_transform(H_all)
            H_c = H_all[:len(H_c)]
            H_d = H_all[len(H_c):]

        if self.method == DirectionMethod.LINEAR_PROBE:
            self.direction_, self.probe_accuracy_ = self._fit_lp(H_c, H_d)
        elif self.method == DirectionMethod.DIFF_OF_MEANS:
            self.direction_ = self._fit_dom(H_c, H_d)
        elif self.method == DirectionMethod.PCA:
            self.direction_ = self._fit_pca(H_c, H_d)

        return self

    def _fit_lp(
        self, H_c: np.ndarray, H_d: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Linear probe: logistic regression, weight vector as direction."""
        X = np.vstack([H_c, H_d])
        y = np.array([1] * len(H_c) + [0] * len(H_d))

        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=self.random_state,
            solver="lbfgs",
        )
        clf.fit(X, y)
        accuracy = clf.score(X, y)

        direction = clf.coef_[0]
        direction = direction / np.linalg.norm(direction)
        return direction, accuracy

    def _fit_dom(self, H_c: np.ndarray, H_d: np.ndarray) -> np.ndarray:
        """Difference of means direction."""
        direction = H_c.mean(axis=0) - H_d.mean(axis=0)
        return direction / np.linalg.norm(direction)

    def _fit_pca(self, H_c: np.ndarray, H_d: np.ndarray) -> np.ndarray:
        """Top PCA component of the pairwise activation difference."""
        from sklearn.decomposition import PCA

        # Pair-wise differences (assumes equal number of samples)
        n = min(len(H_c), len(H_d))
        diffs = H_c[:n] - H_d[:n]

        pca = PCA(n_components=1, random_state=self.random_state)
        pca.fit(diffs)
        direction = pca.components_[0]
        return direction / np.linalg.norm(direction)

    @property
    def direction(self) -> np.ndarray:
        if self.direction_ is None:
            raise RuntimeError("Call .fit() before accessing .direction")
        return self.direction_

    def to_tensor(self) -> torch.Tensor:
        """Return the steering direction as a PyTorch tensor."""
        return torch.tensor(self.direction_, dtype=torch.float32)
