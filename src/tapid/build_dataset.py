"""Dataset construction pipeline (slide 10).

  1. collect clean chunks      -> load_passages()
  2. inject attacks            -> make_example(..., kind="attack")
  3. add hard negatives        -> make_example(..., kind="hard_negative")
  4. perturb (evasion tricks)  -> attacks.perturb()
  5. label                     -> label y in {0, 1} + character span of the injection

Output: JSONL files, one example per line:
  {"id", "task", "chunk", "label", "span": [start, end] | null,
   "attack_type", "perturbation", "source", "lang"}

Usage:
  python -m tapid.build_dataset --source sample --out data/processed
  python -m tapid.build_dataset --source squad --max-passages 10000 --out data/processed
  python -m tapid.build_dataset --source jsonl --passages data/raw/bipia_email.jsonl \
      --text-field context --task-field question \
      --attacks-file data/raw/bipia_attacks.json --test-only --out data/processed/bipia
  python -m tapid.build_dataset --source jsonl --passages data/raw/notinject.jsonl \
      --text-field text --p-attack 0 --p-hard 0 --test-only --out data/processed/notinject
"""

import argparse
import json
import random
import re
from pathlib import Path

from .attacks import HARD_NEGATIVES, perturb, sample_attack

GENERIC_TASKS = [
    "Summarise this passage in two sentences.",
    "What is the main point of this text?",
    "List the key facts mentioned in this document.",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------- step 1
def load_passages(args) -> list[dict]:
    """Return a list of {"task", "text", "source"} dicts."""
    if args.source == "sample":
        path = Path(__file__).resolve().parents[2] / "data" / "sample_passages.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [{"task": r.get("task"), "text": r["text"], "source": "sample"} for r in rows]

    if args.source == "squad":
        from datasets import load_dataset  # needs internet (works on Colab)

        ds = load_dataset("rajpurkar/squad", split="train")
        seen, out = set(), []
        for row in ds:
            if row["context"] in seen:
                continue
            seen.add(row["context"])
            out.append({"task": row["question"], "text": row["context"], "source": "squad"})
            if len(out) >= args.max_passages:
                break
        return out

    if args.source == "jsonl":
        out = []
        for line in Path(args.passages).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out.append({
                "task": row.get(args.task_field) if args.task_field else None,
                "text": row[args.text_field],
                "source": Path(args.passages).stem,
            })
        return out[: args.max_passages]

    raise ValueError(f"unknown source {args.source}")


def load_external_attacks(path: str) -> list[str]:
    """Attack strings from another benchmark (JSON list, or dict of lists)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [a for group in data.values() for a in group]
    return list(data)


# ---------------------------------------------------------------- steps 2-5
def insert_sentence(text: str, sentence: str, rng: random.Random):
    """Insert ``sentence`` between two sentences of ``text``.

    Returns (new_text, (start, end)) where start/end are character offsets
    of the inserted sentence.
    """
    sents = _SENT_SPLIT.split(text.strip())
    pos = rng.randint(0, len(sents))
    before = " ".join(sents[:pos])
    after = " ".join(sents[pos:])
    prefix = before + (" " if before else "")
    new_text = prefix + sentence + (" " + after if after else "")
    return new_text, (len(prefix), len(prefix) + len(sentence))


def make_example(passage, kind, rng, idx, p_perturb=0.2, hinglish=False, external_attacks=None):
    task = passage["task"] or rng.choice(GENERIC_TASKS)
    ex = {
        "id": f"{passage['source']}-{idx}",
        "task": task,
        "label": 0,
        "span": None,
        "attack_type": None,
        "perturbation": None,
        "source": passage["source"],
        "lang": "hinglish" if hinglish else "en",
    }
    if kind == "clean":
        ex["chunk"] = passage["text"]
        return ex
    if kind == "hard_negative":
        ex["chunk"], _ = insert_sentence(passage["text"], rng.choice(HARD_NEGATIVES), rng)
        ex["attack_type"] = "hard_negative"
        return ex

    # kind == "attack"
    if external_attacks:
        atype, sentence = "external", rng.choice(external_attacks)
    else:
        atype, sentence = sample_attack(rng, hinglish=hinglish)
    if rng.random() < p_perturb:
        ex["perturbation"], sentence = perturb(sentence, rng)
    ex["chunk"], span = insert_sentence(passage["text"], sentence, rng)
    ex["label"], ex["span"], ex["attack_type"] = 1, list(span), atype
    return ex


def build(passages, rng, p_attack=0.5, p_hard=0.5, **kw) -> list[dict]:
    """One example per passage: attack with prob p_attack, else a negative
    (hard negative with prob p_hard, otherwise the clean passage)."""
    out = []
    for i, p in enumerate(passages):
        if rng.random() < p_attack:
            kind = "attack"
        else:
            kind = "hard_negative" if rng.random() < p_hard else "clean"
        out.append(make_example(p, kind, rng, i, **kw))
    return out


def split_by_passage(passages, rng, val=0.1, test=0.1):
    """Each passage appears in only one split, so test chunks are truly unseen."""
    idx = list(range(len(passages)))
    rng.shuffle(idx)
    n_val, n_test = int(len(idx) * val), int(len(idx) * test)
    pick = lambda ids: [passages[i] for i in ids]
    return {
        "val": pick(idx[:n_val]),
        "test": pick(idx[n_val:n_val + n_test]),
        "train": pick(idx[n_val + n_test:]),
    }


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["sample", "squad", "jsonl"], default="sample")
    ap.add_argument("--passages", help="JSONL file when --source jsonl")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--task-field", default=None)
    ap.add_argument("--attacks-file", help="use attack strings from another benchmark (test sets only)")
    ap.add_argument("--max-passages", type=int, default=20000)
    ap.add_argument("--copies", type=int, default=1, help="examples generated per passage")
    ap.add_argument("--hinglish", action="store_true", help="use Hinglish attack templates")
    ap.add_argument("--p-attack", type=float, default=0.5, help="share of injected examples")
    ap.add_argument("--p-hard", type=float, default=0.5, help="share of negatives that are hard negatives")
    ap.add_argument("--test-only", action="store_true", help="write everything to test.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    passages = load_passages(args)
    external = load_external_attacks(args.attacks_file) if args.attacks_file else None
    kw = dict(hinglish=args.hinglish, external_attacks=external, p_attack=args.p_attack, p_hard=args.p_hard)

    out = Path(args.out)
    splits = {"test": passages} if args.test_only else split_by_passage(passages, rng)
    for name, ps in splits.items():
        rows = build(ps * args.copies, rng, **kw)
        for i, ex in enumerate(rows):
            ex["id"] = f"{name}-{ex['source']}-{i}"
        write_jsonl(rows, out / f"{name}.jsonl")
        print(f"{name}: {len(rows)} examples ({sum(r['label'] for r in rows)} injected)")


if __name__ == "__main__":
    main()
