"""M4 - Decision module and the public API used by applications.

    p <  tau_1          -> PASS      (chunk unchanged)
    tau_1 <= p < tau_2  -> SANITISE  (remove predicted span S, keep the rest)
    p >= tau_2          -> BLOCK     (drop chunk, log it)

Example:
    guard = InjectionGuard("outputs/proposed")
    result = guard.check("Summarise the refund policy", retrieved_text)
    safe_text = result.text   # what the LLM is allowed to see
"""

from dataclasses import dataclass

import torch

from .data import O, encode
from .model import TaskAwareDetector


@dataclass
class GuardResult:
    action: str          # "pass" | "sanitise" | "block"
    probability: float   # p
    span: str | None     # injected text found by the span head
    text: str            # text to forward to the LLM ("" when blocked)


class InjectionGuard:
    def __init__(self, model_dir, tau1=None, tau2=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer, self.meta = TaskAwareDetector.load(model_dir, self.device)
        self.tau1 = tau1 if tau1 is not None else self.meta["tau1"]
        self.tau2 = tau2 if tau2 is not None else self.meta["tau2"]

    @torch.no_grad()
    def check(self, task: str, chunk: str) -> GuardResult:
        enc = encode(self.tokenizer, task, chunk, None, self.meta["use_task"], self.meta["max_length"])
        ids = torch.tensor([enc["input_ids"]], device=self.device)
        mask = torch.tensor([enc["attention_mask"]], device=self.device)
        out = self.model(ids, mask)
        p = torch.sigmoid(out["cls_logit"])[0].item()
        tags = out["span_logits"].argmax(-1)[0].tolist()

        text = enc["norm_chunk"]
        # with no gold span, token_labels is O on chunk tokens and -100 elsewhere: a chunk-token mask
        start, end = span_from_tags(tags, enc["offsets"], enc["token_labels"])
        span = text[start:end] if start is not None else None

        if p < self.tau1:
            return GuardResult("pass", p, span, text)
        if p < self.tau2 and span:
            return GuardResult("sanitise", p, span, (text[:start] + text[end:]).strip())
        return GuardResult("block", p, span, "")


def span_from_tags(tags, offsets, token_labels):
    """Character range covering every chunk token tagged B or I (None if none)."""
    start = end = None
    for tag, (s, e), lab in zip(tags, offsets, token_labels):
        if lab == -100 or tag == O:  # skip task/special tokens and normal tokens
            continue
        start = s if start is None else min(start, s)
        end = e if end is None else max(end, e)
    return start, end
