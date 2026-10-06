"""S6 (sung2022mat) — C6F6/Ar/O2, O2 유량 스윕을 ICP 와 CCP 에서 각각. 새 모듈 mask_erosion · cd_deviation.

S6 를 넣은 이유: **한 지표만 보면 빠지는 함정**이 있다. CCP 는 마스크가 전혀 깎이지 않아(선택비 ∞)
마스크 침식만 보면 최선인데, 홀이 +31~49% 넓어진다. 저자 최적은 ICP O2 20~22.5 sccm 이다.
ICP 쪽은 O2 가 적으면 마스크가 막혀 식각이 멈추고, 많으면 마스크가 깎이는 트레이드오프다.
"""
import csv
import io as _io
import json
import pathlib
import subprocess
import sys

import pytest

from harc import checker
from harc.defects.base import design_of, show_trend
from harc.defects.cd_deviation import CdDeviation
from harc.defects.mask_erosion import MaskErosion
from harc.defects.registry import ALL

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def series():
    return json.loads((ROOT / "data" / "series" / "S6.json").read_text(encoding="utf-8"))


def _rules(series):
    j = checker.check(series, base_dir=str(ROOT / "data"))["judgeable"]
    return {d.name: d.rule(series, j[d.name]) for d in ALL}


def test_text_values_match_the_paper(series):
    c = {x["id"]: x for x in series["conditions"]}
    # 본문의 CD 변화율 (마스크 홀 160 nm 대비)
    for cid, pct in {"I15": -66, "I20": -25, "I22": -8, "I25": 15, "C150": 31, "C200": 48}.items():
        assert (c[cid]["cd_nm"] / 160 - 1) * 100 == pytest.approx(pct, abs=1), cid
    assert c["I15"]["mask_etch_rate_nm_min"] == 0          # "no etching of ACL was observed" at 15 sccm
    assert c["I25"]["etch_rate_nm_min"] / c["I25"]["mask_etch_rate_nm_min"] == pytest.approx(3.6, abs=0.1)
    assert (c["C150"]["etch_rate_nm_min"], c["C200"]["etch_rate_nm_min"]) == (244, 267)
    assert [k for k, v in c.items() if v["etch_stop"]] == ["I15", "I17"]


def test_figure_read_values_are_declared(series):
    """그림에서 읽은 값은 어느 것인지 기록돼 있어야 한다 — 정확도가 본문 값과 다르다."""
    fr = series["extra"]["figure_read"]
    assert set(fr) == {"etch_rate_nm_min", "mask_etch_rate_nm_min", "cd_nm"}
    assert "I17" in fr["cd_nm"] and "I15" not in fr["cd_nm"]


def test_the_trap_exists(series):
    """마스크 침식만 보면 CCP 가 최선, CD 편차로 보면 ICP 20~22.5 가 최선이어야 한다."""
    r = _rules(series)
    assert r["mask_erosion"]["best_condition"].startswith("C")
    assert r["cd_deviation"]["best_condition"] == "I22"
    ranked = sorted((row["cd_dev_abs_pct"], row["condition"]) for row in r["cd_deviation"]["rows"])
    assert [cid for _, cid in ranked[:2]] == ["I22", "I25"] or ranked[0][1] == "I22"
    assert all(row["cd_dev_side"] == "넓어짐" for row in r["cd_deviation"]["rows"]
               if row["condition"].startswith("C"))


def test_design_is_ofat_and_trend_is_off(series):
    assert design_of(series) == "ofat" and show_trend(series) is False
    for name, out in _rules(series).items():
        assert not [k for k in out if k.startswith("trend")], name


def test_every_condition_has_an_sem_image(series):
    for c in series["conditions"]:
        assert (ROOT / "data" / c["image"]).exists(), c["image"]


def test_answers_are_not_in_the_series_file(series):
    a = json.loads((ROOT / "answers" / "S6.json").read_text(encoding="utf-8"))
    blob = json.dumps({k: v for k, v in series.items() if k != "_note"}, ensure_ascii=False)
    assert a["authors_conclusion"] not in blob
    for phrase in ("최적", "optim", "공정 창", "HARC 식각에 쓸", "삼각형", "테이퍼", "막혀", "막힘", "고질량"):
        assert phrase not in blob, f"시리즈 파일(모델 입력)에 판단이 섞였다: {phrase}"


# ── 새 모듈 ─────────────────────────────────────────────────────────────
def _one(**kw):
    c = {"id": "X", "level": "x"}
    c.update(kw)
    return c


def test_mask_erosion_selectivity_and_deposition():
    s = {"series_id": "T", "conditions": [_one(id="A", mask_etch_rate_nm_min=-50, etch_rate_nm_min=250),
                                          _one(id="B", mask_etch_rate_nm_min=100, etch_rate_nm_min=400)]}
    j = {"A": {"ok": True}, "B": {"ok": True}}
    out = MaskErosion().rule(s, j)
    rows = {r["condition"]: r for r in out["rows"]}
    assert rows["A"]["selectivity"].startswith("∞")
    assert rows["B"]["selectivity"] == 4.0
    assert rows["B"]["change_vs_first"] == "악화" and out["worst_condition"] == "B"


def test_cd_deviation_counts_both_directions_as_bad():
    s = {"series_id": "T", "nominal_cd_nm": 100,
         "conditions": [_one(id="W", cd_nm=130), _one(id="N", cd_nm=70), _one(id="O", cd_nm=102)]}
    j = {k: {"ok": True} for k in "WNO"}
    out = CdDeviation().rule(s, j)
    rows = {r["condition"]: r for r in out["rows"]}
    assert (rows["W"]["cd_dev_pct"], rows["N"]["cd_dev_pct"]) == (30.0, -30.0)
    assert rows["W"]["cd_dev_abs_pct"] == rows["N"]["cd_dev_abs_pct"] == 30.0
    assert out["best_condition"] == "O"


def test_cd_deviation_needs_a_design_cd():
    s = {"series_id": "T", "nominal_cd_nm": None, "conditions": [_one(id="A", top_cd=90)]}
    out = CdDeviation().rule(s, {"A": {"ok": True}})
    assert out["rows"][0]["status"] == "판정 불가" and out["worst_condition"] is None


# ── 루브릭 ─────────────────────────────────────────────────────────────
def _grade(tmp_path, window, onset="I17", issue="cd_deviation"):
    run = {"run_id": "t", "pipeline_version": "v2", "series": "S6", "provider": "mock", "model": None,
           "mode": "single", "rag": False, "images": False,
           "single": {"q1": {"primary_issue": issue, "onset_condition": onset, "process_window": window},
                      "q2": {"hypotheses": []},
                      "q3": {"variable": "O2 유량", "direction": "ICP 로 O2 20~22.5 sccm 을 쓴다"}}}
    (tmp_path / "r.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    out_csv = tmp_path / "o.csv"
    subprocess.run([sys.executable, "-m", "harc.judge", str(tmp_path), "--csv", str(out_csv)],
                   capture_output=True, text=True, cwd=str(ROOT), check=True)
    rows = list(csv.DictReader(_io.open(out_csv, encoding="utf-8-sig")))
    return {r["field"]: r["verdict"] for r in rows if r["verdict"]}


def test_rubric_accepts_the_authors_window(tmp_path):
    v = _grade(tmp_path, "I20 ~ I22 (ICP · O2 20~22.5 sccm)")
    assert v["q1.process_window"] == "correct"
    assert v["q1.primary_issue"] == "correct"
    assert v["q1.onset_condition"] == "correct"
    assert v["q3.direction"] == "correct"


def test_one_end_of_the_window_is_half(tmp_path):
    assert _grade(tmp_path, "I22")["q1.process_window"] == "acceptable"


def test_the_ccp_trap_is_wrong(tmp_path):
    """선택비 ∞ 인 CCP 를 공정 창으로 고르는 것이 이 시리즈가 잡으려는 오답이다."""
    for bad in ("C150 ~ C180", "C200", "I25"):
        assert _grade(tmp_path, bad)["q1.process_window"] == "incorrect", bad


def test_label_matching_does_not_cross_between_tools():
    """'ICP · O2 20 sccm' 이 CCP 조건이나 I22 로 번지면 안 된다 (숫자 라벨 경계, 2026-09-29)."""
    from harc.judge import parse_conditions
    a = json.loads((ROOT / "answers" / "S6.json").read_text(encoding="utf-8"))
    assert parse_conditions("ICP · O2 20 sccm", a["conditions"]) == {"I20"}
    assert parse_conditions("CCP · O2 200 sccm", a["conditions"]) == {"C200"}
