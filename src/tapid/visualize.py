"""Plots and highlighted examples for the Colab notebook.

  plot_history(model_dir)          training loss and validation AUC/F1 per epoch
  plot_roc(model_dir, test_file)   ROC curve with the chosen threshold marked
  highlight_html(result, text)     the chunk with the detected injection in red
"""

import html
import json
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from .data import PairDataset, collate, read_jsonl
from .model import TaskAwareDetector
from .train import predict


def plot_history(model_dir):
    hist = json.loads((Path(model_dir) / "history.json").read_text())
    epochs = [h["epoch"] + 1 for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    a.plot(epochs, [h["train_loss"] for h in hist], marker="o")
    a.set(title="Training loss", xlabel="epoch", ylabel="L = L_cls + λ·L_span")
    for key in ("roc_auc", "f1", "span_f1"):
        if key in hist[0]:
            b.plot(epochs, [h[key] for h in hist], marker="o", label=key)
    b.set(title="Validation scores", xlabel="epoch", ylim=(0, 1.05))
    b.legend()
    fig.tight_layout()
    return fig


def plot_roc(model_dir, test_file, label=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta = TaskAwareDetector.load(model_dir, device)
    rows = read_jsonl(test_file)
    dl = DataLoader(PairDataset(rows, tok, meta["use_task"], meta["max_length"]), batch_size=32,
                    collate_fn=partial(collate, pad_id=tok.pad_token_id))
    probs, _, _ = predict(model, dl, device)
    labels = [r["label"] for r in rows]
    fpr, tpr, thr = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=label or Path(test_file).parent.name)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random guess (AUC 0.5)")
    ax.set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate (recall)")
    ax.legend(loc="lower right")
    return fig


def highlight_html(result, original_text):
    """Show the text with the detected span in red, plus the guard's decision."""
    text = html.escape(original_text)
    if result.span:
        text = text.replace(html.escape(result.span),
                            f"<mark style='background:#ffd5cc'>{html.escape(result.span)}</mark>", 1)
    colour = {"pass": "#2e7d32", "sanitise": "#b26a00", "block": "#c62828"}[result.action]
    return (f"<div style='font-family:sans-serif;max-width:700px'>"
            f"<b style='color:{colour}'>{result.action.upper()}</b> &nbsp; p = {result.probability:.3f}"
            f"<p style='line-height:1.5'>{text}</p></div>")
