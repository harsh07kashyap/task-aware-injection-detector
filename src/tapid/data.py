"""Turns JSONL examples into model inputs.

Input layout (slide 9, M2):
    task-aware    : [CLS] task [SEP] chunk [SEP]
    context-free  : [CLS] chunk [SEP]            (ablation B1)

Token labels for the span head use the BIO scheme on chunk tokens only:
    O = 0 (normal), B = 1 (first token of the injection), I = 2 (inside it).
Task tokens and special tokens get -100 so the loss ignores them.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .normalizer import normalize, normalize_with_span

O, B, I = 0, 1, 2
IGNORE = -100


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def encode(tokenizer, task, chunk, span=None, use_task=True, max_length=512):
    """Tokenise one (task, chunk) pair and build its BIO labels."""
    chunk, span = normalize_with_span(chunk, span)
    task = normalize(task)
    if use_task:
        enc = tokenizer(task, chunk, truncation="only_second", max_length=max_length,
                        return_offsets_mapping=True)
        chunk_seq = 1
    else:
        enc = tokenizer(chunk, truncation=True, max_length=max_length, return_offsets_mapping=True)
        chunk_seq = 0

    seq_ids = enc.sequence_ids()
    labels, prev_inside = [], False
    for (start, end), sid in zip(enc["offset_mapping"], seq_ids):
        if sid != chunk_seq or start == end:
            labels.append(IGNORE)
            continue
        inside = span is not None and start < span[1] and end > span[0]
        labels.append((I if prev_inside else B) if inside else O)
        prev_inside = inside
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_labels": labels,
        "offsets": enc["offset_mapping"],
        "chunk_seq": chunk_seq,
        "norm_chunk": chunk,
    }


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, use_task=True, max_length=512):
        self.rows = rows
        self.items = [
            encode(tokenizer, r["task"], r["chunk"], r.get("span"), use_task, max_length) for r in rows
        ]
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = dict(self.items[i])
        it["label"] = float(self.rows[i]["label"])
        return it


def collate(batch, pad_id=0):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda seq, v: seq + [v] * (n - len(seq))
    return {
        "input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
        "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
        "token_labels": torch.tensor([pad(b["token_labels"], IGNORE) for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch]),
    }
