"""2x2 요인 설계 주효과 (S2: 가스 × 챔버 온도, 지표는 top_bottom_ratio)."""
import pytest
from harc.defects.base import BETTER, WORSE, HIGHER_IS_WORSE
from harc.defects.taper import factorial_effects


def test_s2_main_effects(rules_of):
    f = rules_of("S2")["taper"]["factorial"]
    eff = f["main_effects_pct"]
    assert set(eff) == {"gas", "temp"}
    gas = {e["at"]: e for e in eff["gas"]}          # CHF3 → C3H2F6
    assert gas[20]["change_pct"] == pytest.approx(-25.0, abs=0.1)     # L1 4.0 → L3 3.0
    assert gas[-20]["change_pct"] == pytest.approx(-44.8, abs=0.1)    # L2 2.9 → L4 1.6
    temp = {e["at"]: e for e in eff["temp"]}        # 20C → -20C
    assert temp["CHF3"]["change_pct"] == pytest.approx(-27.5, abs=0.1)     # L1 4.0 → L2 2.9
    assert temp["C3H2F6"]["change_pct"] == pytest.approx(-46.7, abs=0.1)   # L3 3.0 → L4 1.6


def test_main_effect_verdicts_follow_direction(rules_of):
    f = rules_of("S2")["taper"]["factorial"]
    assert f["direction"] == HIGHER_IS_WORSE
    for changes in f["main_effects_pct"].values():
        assert all(e["verdict"] == BETTER for e in changes)   # 네 변화 모두 비율을 낮춘다


def test_factorial_verdict_flips_when_ratio_rises():
    series = {"factors": {"gas": {"A": "g0", "B": "g0", "C": "g1", "D": "g1"},
                          "temp": {"A": 20, "B": -20, "C": 20, "D": -20}}}
    out = factorial_effects(series, {"A": 1.0, "B": 1.0, "C": 2.0, "D": 2.0})
    gas = {e["at"]: e for e in out["main_effects_pct"]["gas"]}
    assert gas[20]["change_pct"] == pytest.approx(100.0)
    assert gas[20]["verdict"] == WORSE
    assert all(e["verdict"] == "변화 없음" for e in out["main_effects_pct"]["temp"])


def test_factorial_needs_a_full_2x2(rules_of):
    series = {"factors": {"gas": {"A": "g0", "B": "g1"}, "temp": {"A": 20, "B": -20}}}
    assert factorial_effects(series, {"A": 1.0, "B": 2.0}) is None   # 칸이 4개가 아니면 계산하지 않는다


def test_single_sweep_series_has_trend_not_factorial(rules_of):
    t = rules_of("S1")["taper"]
    assert "factorial" not in t and "trend_ratio" in t


def test_run_py_exposes_every_pipeline_condition():
    """run.py 가 파이프라인의 실험 조건을 전부 CLI 로 노출하는지.

    조건을 파이프라인에만 추가하고 run.py 에 빠뜨리면 배치로 돌릴 수 없어
    조용히 기본값으로만 실험하게 된다.
    """
    import inspect, pathlib
    from harc import pipeline
    params = set(inspect.signature(pipeline.run).parameters) - {
        "series", "provider", "model", "base_dir"}
    cli = pathlib.Path(pipeline.__file__).parent.parent / "run.py"
    src = cli.read_text(encoding="utf-8")
    flags = {"mode": "--mode", "use_rag": "--rag", "use_images": "--images",
             "cause_hedge": "--cause-hedge", "chunk_chars": "--chunk-chars",
             "temperature": "--temperature", "rule_hints": "--rule-hints",
             "retriever": "--retriever", "chunk_source": "--chunk-source",
             "anonymize_chunks": "--anonymize-chunks", "max_parallel": "--max-parallel"}
    missing = sorted(p for p in params if p not in flags)
    assert not missing, f"run.py 에 플래그가 없는 조건: {missing}"
    for p, flag in flags.items():
        if p in params:
            assert flag in src, f"{p} 의 플래그 {flag} 가 run.py 에 없다"
