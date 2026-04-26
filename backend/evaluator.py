import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

try:
    from backend.config import REPORT_PATH, ensure_dirs
except ImportError:
    from config import REPORT_PATH, ensure_dirs  # type: ignore


def evaluate_predictions(predictions_csv: str = "") -> Dict[str, object]:
    ensure_dirs()
    csv_path = Path(predictions_csv) if predictions_csv else Path(REPORT_PATH) / "predictions_master.csv"
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(
            f"No prediction rows found in {csv_path}. "
            "Check backend/report/prediction_errors.csv for inference failures."
        )

    y_true = (df["actual_label"].str.lower() == "fake").astype(int)
    y_pred = (df["predicted_label"].str.lower() == "fake").astype(int)
    y_score = (df["confidence"].astype(float) / 100.0).clip(0, 1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Per-class metrics computed by treating each class as "positive" in turn.
    real_precision = tn / (tn + fn) if (tn + fn) else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    real_f1 = (
        2 * real_precision * real_recall / (real_precision + real_recall)
        if (real_precision + real_recall)
        else 0.0
    )
    fake_precision = tp / (tp + fp) if (tp + fp) else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) else 0.0
    fake_f1 = (
        2 * fake_precision * fake_recall / (fake_precision + fake_recall)
        if (fake_precision + fake_recall)
        else 0.0
    )

    metrics = {
        "overall_metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "specificity": round(float(specificity), 4),
            "auc_roc": round(float(roc_auc), 4),
        },
        "per_class_metrics": {
            "real": {
                "precision": round(float(real_precision), 4),
                "recall": round(float(real_recall), 4),
                "f1_score": round(float(real_f1), 4),
            },
            "fake": {
                "precision": round(float(fake_precision), 4),
                "recall": round(float(fake_recall), 4),
                "f1_score": round(float(fake_f1), 4),
            },
        },
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_negative": int(tn),
        },
    }

    report_dir = Path(REPORT_PATH)
    with (report_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(report_dir / "roc_curve.png")
    plt.close()

    cm = [[tn, fp], [fn, tp]]
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png")
    plt.close()

    print(f"Saved metrics to {report_dir / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    evaluate_predictions()
