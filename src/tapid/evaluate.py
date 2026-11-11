"""Evaluate a trained detector on one or more test sets (experiments E1-E3).

The threshold tau_2 chosen on validation data is reused unchanged for every
test set, so results reflect deployment behaviour (slide 12 note).

Usage:
  python -m tapid.evaluate --model outputs/proposed \
      --test data/processed/test.jsonl data/processed/bipia/test.jsonl data/processed/notinject/test.jsonl
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import PairDataset, collate, read_jsonl
from .model import TaskAwareDetector
from .train import evaluate_loader


def evaluate_files(model_dir, test_files, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, meta = TaskAwareDetector.load(model_dir, device)
    results = {}
    for path in test_files:
        rows = read_jsonl(path)
        ds = PairDataset(rows, tokenizer, meta["use_task"], meta["max_length"])
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=partial(collate, pad_id=tokenizer.pad_token_id))
        rep = evaluate_loader(model, dl, [r["label"] for r in rows], device, threshold=meta["tau2"])
        rep["n"] = len(rows)
        results[str(path)] = rep
        print(path, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()}))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()
    res = evaluate_files(args.model, args.test)
    out = Path(args.out or Path(args.model) / "test_results.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
