import random

from tapid.build_dataset import build, insert_sentence, split_by_passage
from tapid.normalizer import normalize


def test_insert_sentence_offsets():
    rng = random.Random(0)
    for _ in range(20):
        text, (s, e) = insert_sentence("One. Two. Three.", "Inject me.", rng)
        assert text[s:e] == "Inject me."


def test_labels_and_spans():
    rng = random.Random(1)
    passages = [{"task": "Q?", "text": "Alpha is first. Beta is second. Gamma is third.", "source": "t"}] * 200
    rows = build(passages, rng, p_perturb=0.0)
    attacks = [r for r in rows if r["label"] == 1]
    assert 60 < len(attacks) < 140
    for r in attacks:
        s, e = r["span"]
        assert "ZX-7731" in r["chunk"][s:e]
    assert any(r["attack_type"] == "hard_negative" and r["label"] == 0 for r in rows)


def test_perturbed_attacks_are_recovered_by_normaliser():
    rng = random.Random(2)
    passages = [{"task": "Q?", "text": "Alpha. Beta.", "source": "t"}] * 300
    rows = [r for r in build(passages, rng, p_perturb=1.0) if r["perturbation"] in ("zero_width", "homoglyph")]
    assert rows
    for r in rows:
        assert "ZX-7731" in normalize(r["chunk"]) or "ZX-7731".lower() in normalize(r["chunk"]).lower()


def test_split_has_no_shared_passages():
    rng = random.Random(3)
    passages = [{"text": f"p{i}"} for i in range(100)]
    splits = split_by_passage(passages, rng)
    seen = [p["text"] for rows in splits.values() for p in rows]
    assert len(seen) == len(set(seen)) == 100
