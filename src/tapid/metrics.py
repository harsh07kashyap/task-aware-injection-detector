"""Evaluation metrics (slide 12)."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .data import IGNORE, O


def threshold_for_fpr(labels, scores, target_fpr=0.01) -> float:
    """Highest-recall threshold whose false-positive rate stays <= target_fpr.
    Chosen on validation data, then frozen for every test set."""
    fpr, _, thr = roc_curve(labels, scores)
    ok = np.where(fpr <= target_fpr)[0]
    t = float(thr[ok[-1]])
    return min(t, 1.0)


def classification_report(labels, scores, threshold) -> dict:
    labels, scores = np.asarray(labels).astype(int), np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "threshold": threshold,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if len(set(labels)) == 2:
        out["roc_auc"] = float(roc_auc_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
        out["tpr_at_1pct_fpr"] = float(tpr[fpr <= 0.01].max()) if (fpr <= 0.01).any() else 0.0
    return out


def span_f1(token_labels, token_preds) -> float:
    """Token-level F1 of the injected span: 2|S ∩ Ŝ| / (|S| + |Ŝ|).
    A token counts as 'in the span' when its tag is B or I."""
    gold = pred = both = 0
    for gl, pr in zip(token_labels, token_preds):
        for g, p in zip(gl, pr):
            if g == IGNORE:
                continue
            g_in, p_in = g != O, p != O
            gold += g_in
            pred += p_in
            both += g_in and p_in
    return 2 * both / (gold + pred) if gold + pred else 1.0
