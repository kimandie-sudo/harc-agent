"""S4 (osipov2026acs) — 다목적 최적화 시리즈와 undercut·sidewall_angle 모듈.

S4 를 넣은 이유는 **변별력**이다. S3 는 대조군이 만점(1.000)이라 셀만 늘고
난이도가 늘지 않았다. S4 는 지표마다 최적 조건이 다르고(파레토), 하나만 보고
고르면 틀리도록 설계돼 있다. 그 구조가 유지되는지 고정한다.

또한 이 시리즈는 **스윕이 아니다** — 베이지안 최적화로 입력 6개가 동시에 변한다.
조건 순서가 추세를 뜻하지 않으므로 추세 계산이 꺼져 있어야 한다.
"""
import json
import pathlib
import subprocess
import sys
import csv
import io as _io

import pytest

from harc import checker
from harc.defects.base import design_of, show_trend
from harc.defects.registry import ALL
from harc.defects.undercut import Undercut
from harc.defects.sidewall_angle import SidewallAngle

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def series():
    return json.loads((ROOT / "data" / "series" / "S4.json").read_text(encoding="utf-8"))


@pytest.fixture
def answers():
    return json.loads((ROOT / "answers" / "S4.json").read_text(encoding="utf-8"))


def _by_id(series):
    return {c["id"]: c for c in series["conditions"]}


def test_table_matches_the_paper(series):
    """원문 Table 3 의 14행. 저자 최적 E14 와 각 지표의 단독 최적을 고정한다."""
    c = _by_id(series)
    assert len(c) == 14
    assert (c["E14"]["aspect_ratio"], c["E14"]["undercut_nm"]) == (15.66, 318.4)
    assert min(c.values(), key=lambda x: x["undercut_nm"])["id"] == "E4"
    assert min(c.values(), key=lambda x: x["sidewall_angle_deg"])["id"] == "E7"
    assert max(c.values(), key=lambda x: x["aspect_ratio"])["id"] == "E5"


def test_the_trap_exists(series):
    """**이 시리즈의 존재 이유.** 지표를 하나만 보면 저자 최적과 다른 답이 나와야 한다."""
    c = _by_id(series)
    best_ar = max(c.values(), key=lambda x: x["aspect_ratio"])["id"]
    best_uc = min(c.values(), key=lambda x: x["undercut_nm"])["id"]
    best_sa = min(c.values(), key=lambda x: x["sidewall_angle_deg"])["id"]
    assert len({best_ar, best_uc, best_sa, "E14"}) == 4, \
        "지표별 최적이 겹치면 함정이 사라진다 — 변별력을 잃는다"


def test_single_metric_optima_fail_the_stated_requirement(series):
    """언더컷·측벽각 단독 최적은 원문 요구(AR > 10)를 못 맞춘다. 그래서 오답이다."""
    c = _by_id(series)
    assert series["target_aspect_ratio"] == 10
    for cid in ("E4", "E7", "E11"):
        assert c[cid]["aspect_ratio"] <= 10, f"{cid} 이 요구를 맞추면 오답 근거가 사라진다"
    assert c["E14"]["aspect_ratio"] > 10


def test_design_is_a_sequence_so_trend_is_off(series):
    """입력 6개가 동시에 변한다. 조건 순서를 추세로 읽으면 안 된다."""
    assert design_of(series) == "sequence"
    assert show_trend(series) is False
    j = checker.check(series, base_dir=str(ROOT / "data"))["judgeable"]
    for M in (Undercut, SidewallAngle):
        out = M().rule(series, j[M.name])
        assert not [k for k in out if k.startswith("trend")], \
            f"{M.name} 이 시퀀스 설계에서 추세를 냈다"


def test_sweep_series_still_get_trend():
    """회귀 방지 — sequence 를 도입하면서 기존 스윕 시리즈의 추세를 끄면 안 된다."""
    s1 = json.loads((ROOT / "data" / "series" / "S1.json").read_text(encoding="utf-8"))
    assert design_of(s1) == "sweep" and show_trend(s1) is True


def test_only_the_available_defects_are_judgeable(series):
    j = checker.check(series, base_dir=str(ROOT / "data"))["judgeable"]
    for name in ("undercut", "sidewall_angle", "etch_stop"):
        assert all(v["ok"] for v in j[name].values()), f"{name} 이 판정 불가가 됐다"
    for name in ("bowing", "taper", "necking"):
        assert all(v["ok"] is False for v in j[name].values()), \
            f"{name} 이 판정 가능해졌다 — S4 에는 해당 CD 가 없다"


def test_new_modules_are_registered():
    assert {"undercut", "sidewall_angle"} <= {d.name for d in ALL}


def test_sidewall_angle_compares_by_degrees_not_ratio(series):
    """각도는 0 에 가까울 수 있어 비율 변화가 폭발한다 — 절대 차로 재야 한다."""
    j = checker.check(series, base_dir=str(ROOT / "data"))["judgeable"]
    out = SidewallAngle().rule(series, j["sidewall_angle"])
    rows = {r["condition"]: r for r in out["rows"]}
    assert "change_vs_first_deg" in rows["E7"] and "change_vs_first_pct" not in rows["E7"]
    # E1 1.45° → E7 0.25° 는 1.2도 개선
    assert rows["E7"]["change_vs_first_deg"] == pytest.approx(-1.20)
    assert rows["E7"]["change_vs_first"] == "개선"


def test_defects_read_their_directions(series):
    j = checker.check(series, base_dir=str(ROOT / "data"))["judgeable"]
    uc = Undercut().rule(series, j["undercut"])
    sa = SidewallAngle().rule(series, j["sidewall_angle"])
    assert uc["direction"] == "higher_is_worse"
    assert (uc["worst_condition"], uc["best_condition"]) == ("E13", "E4")
    assert (sa["worst_condition"], sa["best_condition"]) == ("E8", "E7")


def test_answers_are_not_in_the_series_file(series, answers):
    """조건 ID 는 두 파일에 다 있어야 한다. 새면 안 되는 것은 판단과 저자 결론이다."""
    blob = json.dumps(series, ensure_ascii=False)
    assert answers["authors_conclusion"] not in blob
    for key in ("authors_conclusion", "q1", "q2", "q3", "correct", "incorrect", "acceptable"):
        assert f'"{key}"' not in blob, f"시리즈 파일에 루브릭 키: {key}"
    # 판단이 섞였는지는 **데이터 부분만** 본다. _note 는 파일 자체에 대한 메타 주석이라
    # "논문 제목·저자·결론 없음" 같은 문구가 들어가는 것이 정상이다.
    data = json.dumps({k: v for k, v in series.items() if k != "_note"}, ensure_ascii=False)
    for phrase in ("최적 모드", "저자 결론", "파레토", "Pareto", "균형점", "No. 14", "최선", "최악"):
        assert phrase not in data, f"시리즈 파일에 판단이 섞였다: {phrase}"


# ── 루브릭이 세 단계로 갈리는지 ──────────────────────────────────────────
def _grade(tmp_path, aggregate, nxt):
    run = {"run_id": "t", "pipeline_version": "v2", "series": "S4", "provider": "mock",
           "model": None, "mode": "multi", "rag": False, "images": False,
           "aggregate": aggregate,
           "cause": {"hypotheses": "이온 각분포와 불소 라디칼", "cannot_assess": ""},
           "next": nxt}
    (tmp_path / "r.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    out_csv = tmp_path / "o.csv"
    subprocess.run([sys.executable, "-m", "harc.judge", str(tmp_path), "--csv", str(out_csv)],
                   capture_output=True, text=True, cwd=str(ROOT), check=True)
    rows = list(csv.DictReader(_io.open(out_csv, encoding="utf-8-sig")))
    return {r["field"]: r["verdict"] for r in rows if r["verdict"]}


NEXT_OK = {"recommendation": "바이어스 전압을 더 촘촘히 쪼개고 SF6/C4F8 가스비를 조절한다",
           "alternatives": "", "measure_next": ""}
NEXT_BAD = {"recommendation": "압력을 올리고 종횡비만 최대화한다",
            "alternatives": "", "measure_next": ""}


def _agg(issue, window):
    return {"primary_issue": issue, "process_window": window, "ranking": "", "tradeoffs": ""}


def test_rubric_accepts_the_authors_optimum(tmp_path):
    v = _grade(tmp_path, _agg("undercut", "E14 — 균형점"), NEXT_OK)
    for f in ("q1.primary_issue", "q1.process_window", "q3.direction", "q3.variable"):
        assert v.get(f) == "correct", f"{f}: {v}"


def test_rubric_gives_partial_credit_to_a_pareto_alternative(tmp_path):
    """E5 는 파레토 최적이고 AR > 10 을 맞춘다 — 틀리진 않았지만 저자 최적은 아니다."""
    v = _grade(tmp_path, _agg("sidewall_angle", "E5 — 종횡비가 가장 높다"), NEXT_OK)
    assert v.get("q1.process_window") == "acceptable", v
    assert v.get("q1.primary_issue") == "acceptable", v


def test_rubric_rejects_the_single_metric_trap(tmp_path):
    """언더컷만 보고 E4 를 고르는 것이 이 시리즈가 잡으려는 오답이다."""
    v = _grade(tmp_path, _agg("bowing", "E4 — 언더컷이 가장 낮다"), NEXT_BAD)
    for f in ("q1.primary_issue", "q1.process_window", "q3.direction"):
        assert v.get(f) == "incorrect", f"{f}: {v}"


def test_q3_direction_actually_fires(tmp_path):
    """S3 에서는 이 항목이 16건 전부 무채점이었다. 같은 결함을 반복하지 않는다."""
    v = _grade(tmp_path, _agg("undercut", "E14"), NEXT_OK)
    assert "q3.direction" in v, "q3.direction 루브릭이 발화하지 않는다"
