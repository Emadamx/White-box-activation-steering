"""
LinearDeceptionProbe: Trains logistic regression probes at each transformer layer
to find l* — the layer where cooperative vs. deceptive activations are most
linearly separable.

The probe accuracy curve over layers is the primary layer-selection figure.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class LinearDeceptionProbe:
    """
    Trains and evaluates linear probes at each transformer layer.

    Args:
        cv_folds: Number of cross-validation folds for accuracy estimation.
        random_state: Reproducibility seed.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42) -> None:
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.layer_accuracies_: Dict[int, float] = {}
        self.best_layer_: Optional[int] = None

    def fit_all_layers(
        self,
        cooperative_activations: Dict[int, torch.Tensor],
        deceptive_activations: Dict[int, torch.Tensor],
    ) -> "LinearDeceptionProbe":
        """
        Train a probe at every layer and record cross-validated accuracy.

        Args:
            cooperative_activations: {layer_idx: Tensor(n, d)} from cooperative rollouts
            deceptive_activations:   {layer_idx: Tensor(n, d)} from deceptive rollouts

        Returns:
            self
        """
        layer_indices = sorted(
            set(cooperative_activations.keys()) & set(deceptive_activations.keys())
        )

        for layer_idx in layer_indices:
            H_c = cooperative_activations[layer_idx].numpy()
            H_d = deceptive_activations[layer_idx].numpy()

            X = np.vstack([H_c, H_d])
            y = np.array([1] * len(H_c) + [0] * len(H_d))

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
            scores = cross_val_score(clf, X, y, cv=self.cv_folds, scoring="accuracy")
            self.layer_accuracies_[layer_idx] = float(scores.mean())

        self.best_layer_ = max(self.layer_accuracies_, key=self.layer_accuracies_.get)
        return self

    def get_accuracy_curve(self) -> Tuple[List[int], List[float]]:
        """Return (layer_indices, accuracies) for plotting."""
        layers = sorted(self.layer_accuracies_.keys())
        accs = [self.layer_accuracies_[l] for l in layers]
        return layers, accs

    def summary(self) -> str:
        layers, accs = self.get_accuracy_curve()
        lines = ["Layer-wise deception probe accuracy:"]
        for l, a in zip(layers, accs):
            marker = " ← best" if l == self.best_layer_ else ""
            lines.append(f"  Layer {l:2d}: {a:.3f}{marker}")
        return "\n".join(lines)
