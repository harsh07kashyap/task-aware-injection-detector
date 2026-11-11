"""M2 + M3 - pair encoder with a chunk head and a span head.

    h_CLS, h_1..h_n = DeBERTa([CLS] t [SEP] d [SEP])
    p   = sigmoid(w^T h_CLS)            chunk head  (M3a)
    b_i = softmax(V h_i)  over {O,B,I}  span head   (M3b)

    L = L_cls + lambda * L_span
"""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .data import IGNORE


class TaskAwareDetector(nn.Module):
    def __init__(self, backbone: nn.Module, span_weight: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden, 1)    # w
        self.span_head = nn.Linear(hidden, 3)   # V  (O, B, I)
        self.span_weight = span_weight          # lambda

    def forward(self, input_ids, attention_mask, labels=None, token_labels=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(h)
        cls_logit = self.cls_head(h[:, 0]).squeeze(-1)  # w^T h_CLS (sigmoid applied in the loss / at inference)
        span_logits = self.span_head(h)                   # V h_i

        out = {"cls_logit": cls_logit, "span_logits": span_logits}
        if labels is not None:
            l_cls = nn.functional.binary_cross_entropy_with_logits(cls_logit, labels)
            loss = l_cls
            if token_labels is not None and (token_labels != IGNORE).any():
                l_span = nn.functional.cross_entropy(
                    span_logits.reshape(-1, 3), token_labels.reshape(-1), ignore_index=IGNORE
                )
                loss = l_cls + self.span_weight * l_span
                out["l_span"] = l_span.detach()
            out["loss"], out["l_cls"] = loss, l_cls.detach()
        return out

    # ------------------------------------------------------------ save / load
    def save(self, out_dir, tokenizer, extra: dict | None = None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.backbone.config.save_pretrained(out)
        tokenizer.save_pretrained(out)
        torch.save(self.state_dict(), out / "detector.pt")
        meta = {"span_weight": self.span_weight, **(extra or {})}
        (out / "detector.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, model_dir, device="cpu"):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "detector.json").read_text())
        backbone = AutoModel.from_config(AutoConfig.from_pretrained(model_dir))
        model = cls(backbone, span_weight=meta["span_weight"])
        model.load_state_dict(torch.load(model_dir / "detector.pt", map_location=device))
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model.to(device).eval(), tokenizer, meta


def build_model(backbone_name="microsoft/deberta-v3-small", span_weight=0.5):
    """Download the pre-trained encoder and attach both heads."""
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    backbone = AutoModel.from_pretrained(backbone_name)
    return TaskAwareDetector(backbone, span_weight=span_weight), tokenizer
