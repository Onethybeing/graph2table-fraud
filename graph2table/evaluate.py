"""Metrics reported in this project."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def scores(y_true, y_score):
    """AUROC, AUPRC and recall at K (K = number of real frauds), as percentages."""
    k = int(y_true.sum())
    top_k = np.argsort(y_score)[-k:]
    return {
        "AUROC": 100 * roc_auc_score(y_true, y_score),
        "AUPRC": 100 * average_precision_score(y_true, y_score),
        "RecK": 100 * y_true[top_k].sum() / k,
    }


def summarize(rows):
    """Turn a list of per-split metric dicts into {metric: [mean, std]}."""
    return {m: [round(float(np.mean([r[m] for r in rows])), 2), round(float(np.std([r[m] for r in rows])), 2)] for m in rows[0]}
