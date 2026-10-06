"""지표 방향(direction)과 첫 조건 대비 개선/악화 계산."""
import pytest
from harc.defects.base import (verdict_vs_first, best_worst, BETTER, WORSE, SAME,
                               HIGHER_IS_WORSE, LOWER_IS_WORSE, TRUE_IS_WORSE)
from harc.defects.registry import ALL


def test_every_defect_declares_a_direction():
    for d in ALL:
        assert d.metric, f"{d.name}: metric 없음"
        assert d.direction in (HIGHER_IS_WORSE, LOWER_IS_WORSE, TRUE_IS_WORSE)
        assert d.direction_note and d.metric_definition


@pytest.mark.parametrize("cur,ref,direction,expect", [
    (1.6, 4.0, HIGHER_IS_WORSE, BETTER),    # 작아짐 = 개선
    (4.0, 1.6, HIGHER_IS_WORSE, WORSE),
    (1.6, 4.0, LOWER_IS_WORSE, WORSE),      # 방향이 뒤집히면 판정도 뒤집힌다
    (4.0, 1.6, LOWER_IS_WORSE, BETTER),
    (4.0, 4.0, HIGHER_IS_WORSE, SAME),
    (4.02, 4.0, HIGHER_IS_WORSE, SAME),     # 허용 폭(8%) 안
    (None, 4.0, HIGHER_IS_WORSE, None),     # 값 없음은 '변화 없음'이 아니다
])
def test_verdict_vs_first(cur, ref, direction, expect):
    assert verdict_vs_first(cur, ref, direction, tol=0.08) == expect


def test_verdict_for_boolean_metric():
    assert verdict_vs_first(True, False, TRUE_IS_WORSE, relative=False) == WORSE
    assert verdict_vs_first(False, True, TRUE_IS_WORSE, relative=False) == BETTER
    assert verdict_vs_first(False, False, TRUE_IS_WORSE, relative=False) == SAME


def test_best_worst_ignores_missing_and_breaks_ties_by_order():
    bw = best_worst([("A", 2.0), ("B", None), ("C", 2.0), ("D", 1.0)], HIGHER_IS_WORSE)
    assert bw["worst_condition"] == "A" and bw["best_condition"] == "D"


# --- 실제 시리즈: S2 는 L4 가 가장 수직(최선), L1 이 가장 심함(최악) ---

def test_s2_taper_best_is_L4(rules_of):
    t = rules_of("S2")["taper"]
    assert t["direction"] == HIGHER_IS_WORSE
    assert t["best_condition"] == "L4" and t["best_value"] == 1.6
    assert t["worst_condition"] == "L1" and t["worst_value"] == 4.0
    assert t["summary"] == "최악 L1(4.0), 최선 L4(1.6), 값이 작을수록 수직"


def test_s2_taper_rows_improve_vs_first(rules_of):
    rows = {r["condition"]: r for r in rules_of("S2")["taper"]["rows"]}
    assert rows["L1"]["change_vs_first"] == SAME          # 기준 조건
    assert rows["L2"]["change_vs_first"] == BETTER
    assert rows["L3"]["change_vs_first"] == BETTER
    assert rows["L4"]["change_vs_first"] == BETTER
    assert rows["L4"]["change_vs_first_pct"] == -60.0


def test_s1_bowing_direction(rules_of):
    b = rules_of("S1")["bowing"]
    rows = {r["condition"]: r for r in b["rows"]}
    assert rows["C1"]["bow_index_pct"] == pytest.approx(12.2, abs=0.1)   # 221/197
    assert rows["C2"]["bow_index_pct"] == pytest.approx(-1.5, abs=0.1)   # 194/197
    assert rows["C3"]["bow_index_pct"] == pytest.approx(-4.6, abs=0.1)   # 187/196
    assert rows["C2"]["change_vs_first"] == BETTER        # 12.2% → -1.5%
    assert rows["C2"]["change_vs_first_pp"] == pytest.approx(-13.7, abs=0.1)
    # 보잉 지수만 보면 최선은 C3 다. 논문이 최적으로 꼽은 조건은 C2 지만 그것은
    # 보잉과 좁아짐을 함께 본 판단이고(answers/S1.json 참고), 이 규칙은 보잉만 잰다.
    assert b["worst_condition"] == "C1" and b["best_condition"] == "C3"
    # C4·C5 는 원문 본문에 max_cd 가 없다 (Figure 4 그래프에만) → 추측 금지
    assert rows["C4"].get("status") == "판정 불가"
    assert rows["C5"].get("status") == "판정 불가"


def test_s1_etch_stop_direction(rules_of):
    e = rules_of("S1")["etch_stop"]
    rows = {r["condition"]: r for r in e["rows"]}
    assert rows["C1"]["change_vs_first"] == SAME
    assert rows["C4"]["change_vs_first"] == WORSE
    assert e["worst_condition"] == "C4" and e["best_condition"] == "C1"
    assert e["onset"] == "C4"
