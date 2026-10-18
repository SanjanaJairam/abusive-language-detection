"""Evaluation: the original weighted metrics plus the per-class view that weighted averages hide."""
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from .config import CLASSES


def evaluate(y_true, y_pred):
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class": {
            name: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(support[i])}
            for i, name in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
