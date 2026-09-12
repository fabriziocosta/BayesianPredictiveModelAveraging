"""Small helpers shared by the question notebooks.

The notebooks deliberately keep the experimental loops visible. This module only
contains repeatable data generation, fitting, and metric helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import make_classification, make_moons, make_regression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

from bayesian_predictive_model_averaging import (
    BayesianPredictiveModelAveragingClassifier,
    BayesianPredictiveModelAveragingRegressor,
)


def classification_data(seed: int = 7, kind: str = "nonlinear") -> tuple[np.ndarray, np.ndarray]:
    """Return a compact binary benchmark suitable for notebook iteration."""

    if kind == "moons":
        return make_moons(n_samples=600, noise=0.25, random_state=seed)
    if kind == "linear":
        return make_classification(
            n_samples=600,
            n_features=8,
            n_informative=5,
            n_redundant=1,
            class_sep=1.2,
            random_state=seed,
        )
    return make_classification(
        n_samples=600,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        class_sep=0.8,
        flip_y=0.08,
        random_state=seed,
    )


def regression_data(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Return a nonlinear-enough regression benchmark for quick ablations."""

    X, y = make_regression(
        n_samples=600,
        n_features=8,
        n_informative=6,
        noise=12.0,
        random_state=seed,
    )
    return X, y


def split_data(
    X: np.ndarray, y: np.ndarray, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratify = y if len(np.unique(y)) < 10 else None
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=stratify)


def fit_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 7,
    **kwargs: Any,
) -> BayesianPredictiveModelAveragingClassifier:
    options = {"n_estimators": 8, "cv": 3, "n_jobs": 1, "random_state": seed}
    options.update(kwargs)
    model = BayesianPredictiveModelAveragingClassifier(**options)
    model.fit(X_train, y_train)
    return model


def fit_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 7,
    **kwargs: Any,
) -> BayesianPredictiveModelAveragingRegressor:
    options = {"n_estimators": 8, "cv": 3, "n_jobs": 1, "random_state": seed}
    options.update(kwargs)
    model = BayesianPredictiveModelAveragingRegressor(**options)
    model.fit(X_train, y_train)
    return model


def classification_metrics(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(X_test)
    return {
        "log_loss": float(log_loss(y_test, probabilities)),
        "brier": float(brier_score_loss(y_test, probabilities[:, 1])),
        "accuracy": float(accuracy_score(y_test, np.argmax(probabilities, axis=1))),
        "max_probability": float(np.max(probabilities, axis=1).mean()),
    }


def regression_metrics(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
    }


def model_diagnostics(model: Any) -> dict[str, Any]:
    return {
        "n_estimators": int(model.n_estimators_),
        "ess": getattr(model, "effective_sample_size_", None),
        "ess_fraction": getattr(model, "effective_sample_size_fraction_", None),
        "family_mass": model.get_model_masses()["family"],
    }
