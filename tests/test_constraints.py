"""v5: 설계 요구(하드 제약)는 규칙이 판정한다 (harc/constraints.py).

v4 에서 S4 8런 중 5런이 "AR > 10 을 만족하지 못한다" 고 적으면서 그 조건을 공정 창으로 골랐다.
"""
import json
import pathlib

import pytest

from harc import constraints as hard
from harc import pipeline


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    real = pipeline.call_json
    def spy(agent, provider, system, user, *a, **kw):
        calls.append({"agent": agent, "system": system, "user": user})
        return real(agent, provider, system, user, *a, **kw)
    monkeypatch.setattr(pipeline, "call_json", spy)
    return calls

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _s4():
    return json.loads((ROOT / "data" / "series" / "S4.json").read_text(encoding="utf-8"))


def test_s4_eligible_set_is_the_ar_over_10_conditions():
    r = hard.evaluate(_s4())
    assert set(r["eligible"]) == {"E2", "E3", "E5", "E6", "E9", "E14"}
    assert {"E4", "E7", "E11"} <= set(r["excluded"]) and "aspect_ratio = 7.89" in r["excluded"]["E4"][0]
    assert r["unknown"] == []


def test_series_without_constraints_are_untouched():
    for sid in ("S1", "S5", "S8"):
        s = json.loads((ROOT / "data" / "series" / f"{sid}.json").read_text(encoding="utf-8"))
        assert hard.evaluate(s) is None and hard.prompt_block(None) == ""


def test_violations_are_flagged_but_the_answer_is_not_changed():
    s, r = _s4(), hard.evaluate(_s4())
    assert hard.check("E4 (언더컷 최소)", s, r) == {"chosen": ["E4"], "violations": ["E4"]}
    assert hard.check(["E14"], s, r)["violations"] == []
    assert hard.check("E1, E14", s, r)["chosen"] == ["E1", "E14"]     # E1 이 E14 안에서 잡히면 안 된다


def test_constraint_block_reaches_both_modes(recorded):
    s = _s4()
    out = pipeline.run(s, provider="mock", mode="multi", base_dir="data")
    agg = next(c for c in recorded if c["agent"] == "aggregate")
    assert "설계 요구 판정" in agg["user"] and "E14" in agg["user"]
    assert out["constraints"]["eligible"] and "constraint_check" in out
    recorded.clear()
    out = pipeline.run(s, provider="mock", mode="single", base_dir="data")
    single = next(c for c in recorded if c["agent"] == "single")
    assert "설계 요구 판정" in single["user"] and "constraint_check" in out


def test_s4_pareto_front_uses_the_papers_four_objectives():
    """목표 넷(종횡비·언더컷·측벽각·등방성)으로 판정 — E9 만 지배당한다.
    셋만 쓰면 등방성이 가장 좋은 E3 가 E14 에 지배당한 것으로 잘못 찍힌다."""
    r = hard.evaluate(_s4())
    assert r["dominated"].keys() == {"E9"} and "E6" in r["dominated"]["E9"]
    assert set(r["pareto_front"]) == {"E2", "E3", "E5", "E6", "E14"}
    assert set(r["candidate_values"]["E3"]) == {"종횡비", "언더컷", "측벽각 이탈", "식각 등방성"}
    assert "dominated" in hard.prompt_block(r) and "통째로" in hard.prompt_block(r)


def test_pareto_drops_an_objective_missing_for_some_candidate():
    s = {"constraints": [{"field": "a", "op": ">", "value": 0}],
         "objectives": [{"field": "a", "better": "higher"}, {"field": "b", "better": "lower"}],
         "conditions": [{"id": "X", "a": 2, "b": 1}, {"id": "Y", "a": 1, "b": None}]}
    r = hard.evaluate(s)
    assert r["objectives"] == ["a(클수록 좋음)"] and r["dominated"] == {"Y": ["X"]}
