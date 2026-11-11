"""Fine-tune the detector:  L = L_cls + lambda * L_span.

Usage (Colab T4):
  python -m tapid.train --config configs/default.yaml
  python -m tapid.train --config configs/default.yaml --no-task --out outputs/b1_context_free   # ablation B1
"""

import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data import PairDataset, collate, read_jsonl
from .metrics import classification_report, span_f1, threshold_for_fpr
from .model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, device):
    """Return injection probabilities p, gold/pred token tags per example."""
    model.eval()
    probs, golds, preds = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], batch["attention_mask"])
        probs += torch.sigmoid(out["cls_logit"]).float().cpu().tolist()
        tags = out["span_logits"].argmax(-1).cpu()
        for gl, pr in zip(batch["token_labels"].cpu(), tags):
            golds.append(gl.tolist())
            preds.append(pr.tolist())
    return probs, golds, preds


def evaluate_loader(model, loader, labels, device, threshold=None, target_fpr=0.01):
    probs, golds, preds = predict(model, loader, device)
    if threshold is None:
        threshold = threshold_for_fpr(labels, probs, target_fpr) if len(set(labels)) == 2 else 0.5
    report = classification_report(labels, probs, threshold)
    report["span_f1"] = span_f1(golds, preds)
    return report


def train(cfg, model=None, tokenizer=None):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model, tokenizer = build_model(cfg["backbone"], cfg["span_weight"])
    model.to(device)

    mk = lambda path: PairDataset(read_jsonl(path), tokenizer, cfg["use_task"], cfg["max_length"])
    train_ds, val_ds = mk(cfg["train_file"]), mk(cfg["val_file"])
    col = partial(collate, pad_id=tokenizer.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=col)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2, collate_fn=col)
    val_labels = [r["label"] for r in val_ds.rows]

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = len(train_dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    use_amp = device == "cuda" and cfg.get("fp16", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, history = -1.0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, enabled=use_amp):
                out = model(**batch)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()
            total += out["loss"].item()
            if step % cfg["log_every"] == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} loss {out['loss'].item():.4f}")

        report = evaluate_loader(model, val_dl, val_labels, device, target_fpr=cfg["target_fpr"])
        report.update(epoch=epoch, train_loss=total / max(len(train_dl), 1))
        history.append(report)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in report.items()}))

        score = report.get("roc_auc", report["f1"])
        if score > best:
            best = score
            # tau_2 = threshold at the target FPR; tau_1 a lower "sanitise" threshold
            tau2 = report["threshold"]
            model.save(cfg["out_dir"], tokenizer, {
                "use_task": cfg["use_task"], "max_length": cfg["max_length"],
                "tau1": round(tau2 * cfg["tau1_ratio"], 4), "tau2": tau2,
                "backbone": cfg["backbone"], "val_report": report,
            })
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["out_dir"]) / "history.json").write_text(json.dumps(history, indent=2))
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-task", action="store_true", help="context-free ablation (B1)")
    ap.add_argument("--out", help="override out_dir")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.no_task:
        cfg["use_task"] = False
    if args.out:
        cfg["out_dir"] = args.out
    train(cfg)


if __name__ == "__main__":
    main()
