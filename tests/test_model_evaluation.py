"""Tests for the Model Evaluation gate (champion vs challenger on the test set)."""

from types import SimpleNamespace

import pandas as pd

from phishsentinel.components import model_evaluation as me
from phishsentinel.components.model_evaluation import ModelEvaluation


class _FakeModel:
    def __init__(self, preds):
        self._preds = preds

    def predict(self, x):
        return self._preds[: len(x)]


def _make_test_csv(tmp_path):
    # y after mapping -1 -> 0 becomes [1, 1, 0, 0]
    df = pd.DataFrame({"f1": [1, 1, -1, -1], "Result": [1, 1, -1, -1]})
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_promotes_when_candidate_beats_champion(tmp_path, monkeypatch):
    promoted = {}
    monkeypatch.setattr(me, "load_model_version", lambda v: _FakeModel([1, 1, 0, 0]))  # perfect
    monkeypatch.setattr(me, "load_production_model", lambda: _FakeModel([1, 1, 1, 1]))  # worse
    monkeypatch.setattr(me, "set_production", lambda v, f1=None: promoted.update(v=v) or True)

    cfg = SimpleNamespace(min_improvement=0.0)
    art = ModelEvaluation(cfg, _make_test_csv(tmp_path), candidate_version="2").initiate_model_evaluation()

    assert art.is_model_accepted is True
    assert art.promoted_version == "2"
    assert promoted["v"] == "2"


def test_rejects_when_candidate_worse(tmp_path, monkeypatch):
    promoted = {}
    monkeypatch.setattr(me, "load_model_version", lambda v: _FakeModel([1, 1, 1, 1]))  # worse
    monkeypatch.setattr(me, "load_production_model", lambda: _FakeModel([1, 1, 0, 0]))  # perfect
    monkeypatch.setattr(me, "set_production", lambda v, f1=None: promoted.update(v=v) or True)

    cfg = SimpleNamespace(min_improvement=0.0)
    art = ModelEvaluation(cfg, _make_test_csv(tmp_path), candidate_version="2").initiate_model_evaluation()

    assert art.is_model_accepted is False
    assert art.promoted_version is None
    assert promoted == {}  # set_production must NOT be called


def test_accepts_first_model_when_no_champion(tmp_path, monkeypatch):
    monkeypatch.setattr(me, "load_model_version", lambda v: _FakeModel([1, 1, 0, 0]))
    monkeypatch.setattr(me, "load_production_model", lambda: None)
    monkeypatch.setattr(me, "set_production", lambda v, f1=None: True)

    cfg = SimpleNamespace(min_improvement=0.0)
    art = ModelEvaluation(cfg, _make_test_csv(tmp_path), candidate_version="1").initiate_model_evaluation()

    assert art.is_model_accepted is True
    assert art.champion_f1 == -1.0
