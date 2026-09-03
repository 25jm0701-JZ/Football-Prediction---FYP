"""
XGBoost and Random Forest wrappers compatible with the SoftmaxModel interface.

Each class exposes ``fit(features, targets) → self`` and
``predict_proba(features) → np.ndarray`` so they can be dropped into
the existing cross-validation pipeline.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


@dataclass
class XGBoostModel:
    """Multinomial XGBoost classifier matching SoftmaxModel's interface."""

    params: dict[str, Any] = field(default_factory=lambda: {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "seed": 42,
        "verbosity": 0,
    })
    n_estimators: int = 400
    early_stopping_rounds: int = 20
    _model: Any = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "XGBoostModel":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed.")
        labels = targets.argmax(axis=1)
        dtrain = xgb.DMatrix(features, label=labels)

        evals_result: dict[str, dict[str, list[float]]] = {}
        evals = [(dtrain, "train")]
        if eval_set is not None:
            dval = xgb.DMatrix(eval_set[0], label=eval_set[1].argmax(axis=1))
            evals.append((dval, "eval"))

        self._model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            evals_result=evals_result,
            verbose_eval=False,
        )
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        dtest = xgb.DMatrix(features)
        raw = self._model.predict(dtest)
        if raw.ndim == 1:
            raw = raw.reshape(-1, self.params["num_class"])
        return raw

    def to_dict(self) -> dict:
        if self._model is None:
            raise RuntimeError("Cannot serialize an unfitted XGBoost model.")
        raw_bytes = self._model.save_raw()
        if isinstance(raw_bytes, memoryview):
            raw_bytes = bytes(raw_bytes)
        return {
            "model_type": "xgboost",
            "model_bytes_b64": base64.b64encode(raw_bytes).decode("ascii"),
            "params": {k: v for k, v in self.params.items()},
            "num_class": self.params.get("num_class", 3),
            "n_estimators": self.n_estimators,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "XGBoostModel":
        raw = base64.b64decode(payload["model_bytes_b64"])
        model = cls(
            params=payload.get("params", {}),
            n_estimators=payload.get("n_estimators", 400),
        )
        model._model = xgb.Booster(model_file=io.BytesIO(raw))
        return model


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

try:
    from sklearn.ensemble import RandomForestClassifier

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class RFModel:
    """Random Forest classifier matching SoftmaxModel's interface."""

    params: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    })
    _model: Any = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        **kwargs: Any,
    ) -> "RFModel":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn is not installed.")
        labels = targets.argmax(axis=1)
        self._model = RandomForestClassifier(**self.params)
        self._model.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        probs = self._model.predict_proba(features)
        # sklearn returns list of (n, n_classes) per class; stack
        if isinstance(probs, list):
            return np.column_stack(probs)
        return np.asarray(probs)


# ---------------------------------------------------------------------------
# Gradient Boosting (sklearn)
# ---------------------------------------------------------------------------


@dataclass
class GBModel:
    """Gradient Boosting classifier matching SoftmaxModel's interface."""

    params: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 3,
        "min_samples_leaf": 10,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": 42,
    })
    _model: Any = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        **kwargs: Any,
    ) -> "GBModel":
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn is not installed.")
        from sklearn.ensemble import GradientBoostingClassifier

        labels = targets.argmax(axis=1)
        self._model = GradientBoostingClassifier(**self.params)
        self._model.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return np.asarray(self._model.predict_proba(features))
