"""Shared fixtures: a tiny, randomly initialised DeBERTa and a small tokenizer.

They let the whole pipeline run offline on a CPU in seconds. Real training
uses microsoft/deberta-v3-small instead (configs/default.yaml).
"""

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import AutoModel, DebertaV2Config, PreTrainedTokenizerFast

from tapid.attacks import ATTACKS, HARD_NEGATIVES, HINGLISH_ATTACKS
from tapid.model import TaskAwareDetector

ROOT = Path(__file__).resolve().parents[1]


def _corpus():
    rows = [json.loads(l) for l in (ROOT / "data" / "sample_passages.jsonl").read_text().splitlines()]
    texts = [r["text"] for r in rows] + [r["task"] for r in rows]
    texts += [a for group in ATTACKS.values() for a in group] + HARD_NEGATIVES + HINGLISH_ATTACKS
    return texts


@pytest.fixture(scope="session")
def tiny_tokenizer():
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(_corpus(), trainers.WordPieceTrainer(vocab_size=800, special_tokens=special))
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


@pytest.fixture()
def tiny_model(tiny_tokenizer):
    cfg = DebertaV2Config(vocab_size=len(tiny_tokenizer), hidden_size=64, num_hidden_layers=2,
                          num_attention_heads=2, intermediate_size=128, max_position_embeddings=512,
                          relative_attention=True, position_buckets=64, pos_att_type=["p2c", "c2p"],
                          type_vocab_size=0)
    return TaskAwareDetector(AutoModel.from_config(cfg), span_weight=0.5)
