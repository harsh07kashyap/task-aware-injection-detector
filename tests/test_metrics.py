import pytest

from tapid.metrics import classification_report, span_f1, threshold_for_fpr


def test_perfect_separation():
    rep = classification_report([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert rep["f1"] == 1.0 and rep["fpr"] == 0.0 and rep["roc_auc"] == 1.0


def test_threshold_respects_fpr():
    labels = [0] * 100 + [1] * 100
    scores = [i / 100 for i in range(100)] + [0.5 + i / 200 for i in range(100)]
    t = threshold_for_fpr(labels, scores, 0.01)
    rep = classification_report(labels, scores, t)
    assert rep["fpr"] <= 0.01


def test_span_f1():
    gold = [[-100, 0, 1, 2, 0]]
    pred = [[0, 0, 1, 0, 0]]      # found 1 of 2 span tokens; position 0 is ignored
    assert span_f1(gold, pred) == pytest.approx(2 * 1 / (2 + 1))
