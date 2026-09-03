"""
Detailed evaluation metrics and visualization for football predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _resolve_class_names(y_true: np.ndarray, class_names: Optional[list[str]] = None) -> list[str]:
    """Auto-detect class names based on number of classes."""
    if class_names is not None:
        return class_names
    n_classes = len(np.unique(y_true))
    if n_classes == 3:
        return ["Away", "Draw", "Home"]
    elif n_classes == 2:
        return ["Under", "Over"]
    return [str(i) for i in range(n_classes)]


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels (0, 1, 2 for A/D/H)
        y_pred: Predicted labels
        y_prob: Predicted probabilities (n_samples, n_classes)
        class_names: Names for each class (auto-detected if None)

    Returns:
        Dictionary of metrics
    """
    class_names = _resolve_class_names(y_true, class_names)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class metrics
    per_class = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        per_class[name] = {
            "support": int(mask.sum()),
            "precision": float(precision_score(y_true == i, y_pred == i, zero_division=0)),
            "recall": float(recall_score(y_true == i, y_pred == i, zero_division=0)),
            "f1": float(f1_score(y_true == i, y_pred == i, zero_division=0)),
        }
    metrics["per_class"] = per_class

    # Probability-based metrics
    if y_prob is not None:
        n_classes = y_prob.shape[1]
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except Exception:
            metrics["log_loss"] = None

        # Brier score per class
        brier_scores = []
        for i in range(n_classes):
            y_binary = (y_true == i).astype(int)
            bs = brier_score_loss(y_binary, y_prob[:, i])
            brier_scores.append(float(bs))
        metrics["brier_per_class"] = brier_scores
        metrics["brier_mean"] = float(np.mean(brier_scores))

        # ROC AUC (one-vs-rest)
        try:
            auc_scores = []
            for i in range(n_classes):
                y_binary = (y_true == i).astype(int)
                if len(np.unique(y_binary)) > 1:  # need both classes
                    auc = roc_auc_score(y_binary, y_prob[:, i])
                    auc_scores.append(float(auc))
            metrics["roc_auc_per_class"] = auc_scores
            metrics["roc_auc_macro"] = float(np.mean(auc_scores))
        except Exception:
            metrics["roc_auc_macro"] = None

    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Results") -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    print(f"\nOverall:")
    for key in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")

    if "log_loss" in metrics and metrics["log_loss"] is not None:
        print(f"  log_loss: {metrics['log_loss']:.4f}")
    if "brier_mean" in metrics:
        print(f"  brier_score (mean): {metrics['brier_mean']:.4f}")
    if "roc_auc_macro" in metrics and metrics["roc_auc_macro"] is not None:
        print(f"  roc_auc (macro): {metrics['roc_auc_macro']:.4f}")

    print(f"\nPer-Class:")
    print(f"  {'Class':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Support':<8}")
    for name, cm in metrics.get("per_class", {}).items():
        print(f"  {name:<8} {cm['precision']:<8.3f} {cm['recall']:<8.3f} "
              f"{cm['f1']:<8.3f} {cm['support']:<8d}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot confusion matrix."""
    class_names = _resolve_class_names(y_true, class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=class_names,
        cmap="Blues",
        normalize="true",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Normalized Confusion Matrix")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[list[str]] = None,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot predicted probability distributions by true class."""
    class_names = _resolve_class_names(y_true, class_names)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, (ax, name) in enumerate(zip(axes, class_names)):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        for j, cls_name in enumerate(class_names):
            ax.hist(
                y_prob[mask, j],
                bins=10,
                alpha=0.5,
                label=cls_name,
                range=(0, 1),
            )
        ax.set_title(f"True: {name} (n={mask.sum()})")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[list[str]] = None,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot reliability diagram (calibration curve)."""
    class_names = _resolve_class_names(y_true, class_names)

    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, name in enumerate(class_names):
        y_binary = (y_true == i).astype(int)
        prob_pred, prob_true = calibration_curve(y_binary, y_prob[:, i], n_bins=5)
        ax.plot(prob_pred, prob_true, marker="o", label=name)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def save_predictions(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    save_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Save predictions with match info for inspection."""
    result = df_test[["date", "home_team", "away_team", "home_goals_full",
                       "away_goals_full", "result_full"]].copy()

    # Map numeric labels to readable ones
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    if n_classes == 3:
        label_map = {0: "A", 1: "D", 2: "H"}
    elif n_classes == 2:
        label_map = {0: "Under", 1: "Over"}
    else:
        label_map = {i: str(i) for i in range(n_classes)}

    result["actual"] = [label_map.get(int(v), v) for v in y_true]
    result["predicted"] = [label_map.get(int(v), v) for v in y_pred]

    if y_prob is not None:
        n_classes = y_prob.shape[1]
        if n_classes == 3:
            # Output columns in H/D/A order (index 2=Home, 1=Draw, 0=Away)
            cls_order = [("prob_home", 2), ("prob_draw", 1), ("prob_away", 0)]
        elif n_classes == 2:
            cls_order = [("prob_over", 1), ("prob_under", 0)]
        else:
            cls_order = [(f"prob_class_{i}", i) for i in range(n_classes)]
        for name, idx in cls_order:
            result[name] = y_prob[:, idx]

    result["correct"] = (y_true == y_pred).astype(int)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)
        print(f"Saved predictions: {save_path}")

    return result


def evaluate_overtime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.Series,
    window: str = "30D",
) -> pd.DataFrame:
    """Evaluate accuracy over time using a rolling window."""
    df = pd.DataFrame({"date": pd.to_datetime(dates), "correct": (y_true == y_pred).astype(int)})
    df = df.sort_values("date").reset_index(drop=True)
    df["rolling_accuracy"] = df["correct"].rolling(window, min_periods=5).mean()
    return df[["date", "correct", "rolling_accuracy"]]
