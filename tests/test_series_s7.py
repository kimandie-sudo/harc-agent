"""S7 (kimlee2022mnsl) — Si 마이크로 홀 DRIE, 한 번에 한 변수씩 네 스윕. 새 모듈 depth_loss.

S7 를 넣은 이유: 저자가 고른 값이 **지표의 최솟값이 아니다**. 바이어스 400 V·C4F8 80 sccm 은 면적 감소비가
300 V·70 sccm 과 같거나 0.1~0.3%p 낮지만 저자는 "더 나아지지 않았다"며 앞의 값을 골랐다. ICP 파워는
면적 감소비를 줄이는 대신 식각 깊이를 깎는다. 포화점과 트레이드오프를 읽는 판단이 필요한 시리즈다.
"""
import csv
import io as _io
import json
import math
import pathlib
import subprocess
import sys

import pytest

from harc import checker
from harc.defects.base import design_of, show_trend
from harc.defects.depth_loss import DepthLoss
from harc.defects.registry import ALL

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def series():
    return json.loads((ROOT / "data" / "series" / "S7.json").read_text(encoding="utf-8"))


def _rules(series):
    j = checker.check(series, base_dir=str(ROOT / "data"))["judgeable"]
    return {d.name: d.rule(series, j[d.name]) for d in ALL}


def _c(series):
    return {x["id"]: x for x in series["conditions"]}


def test_figure_values_match_the_text(series):
    """그래프에서 읽은 값을 본문 수치와 본문의 개선 폭(35·12·7·8%p)으로 검증한다."""
    c = _c(series)
    arr = {k: v["area_reduction_ratio_pct"] for k, v in c.items()}
    for cid, pct in {"B10": 91, "B300": 56, "P1500": 50, "C70": 55, "S120": 69}.items():
        assert arr[cid] == pytest.approx(pct, abs=0.6), cid
    assert arr["B10"] - arr["B300"] == pytest.approx(35, abs=1)     # 바이어스
    assert arr["B200"] - arr["P1500"] == pytest.approx(12, abs=1)   # ICP (출발점 = B200)
    assert arr["B200"] - arr["C70"] == pytest.approx(7, abs=1)      # C4F8
    assert arr["S150"] - arr["S120"] == pytest.approx(8, abs=1)     # SF6


def test_top_bottom_ratio_follows_the_area_definition(series):
    """ARR = 1 − (DB/DT)² 이므로 DT/DB = 1/√(1 − ARR)."""
    for c in series["conditions"]:
        a = c["area_reduction_ratio_pct"] / 100
        assert c["top_bottom_ratio"] == pytest.approx(1 / math.sqrt(1 - a), abs=1e-3), c["id"]


def test_shared_center_point_is_not_duplicated(series):
    """ICP 1200 W · C4F8 60 sccm 은 바이어스 200 V 와 같은 시료다 (같은 SEM 사진) — 따로 두지 않는다."""
    ids = set(_c(series))
    assert not ids & {"P1200", "C60"}
    b = _c(series)["B200"]
    assert (b["bias_v"], b["icp_w"], b["c4f8_sccm"], b["sf6_sccm"]) == (200, 1200, 60, 150)
    for cid in ("P1500", "P1700", "C70", "C80"):
        assert _c(series)[cid]["bias_v"] == 200, cid


def test_figure_read_values_are_declared(series):
    fr = series["extra"]["figure_read"]
    assert set(fr) == {"area_reduction_ratio_pct", "etch_depth_um"}
    assert set(fr["etch_depth_um"]) == set(_c(series))


def test_authors_pick_is_not_the_metric_minimum(series):
    """이 시리즈가 잡으려는 판단: 규칙의 최선과 저자의 선택이 다르다."""
    arr = {k: v["area_reduction_ratio_pct"] for k, v in _c(series).items()}
    assert arr["B400"] <= arr["B300"] and arr["C80"] <= arr["C70"]
    r = _rules(series)
    assert r["taper"]["worst_condition"] == "B10"
    assert r["depth_loss"]["worst_condition"] == "P1700"


def test_design_is_ofat_and_trend_is_off(series):
    assert design_of(series) == "ofat" and show_trend(series) is False
    for name, out in _rules(series).items():
        assert not [k for k in out if k.startswith("trend")], name


def test_listed_sem_images_exist(series):
    for c in series["conditions"]:
        if c.get("image"):
            assert (ROOT / "data" / c["image"]).exists(), c["image"]
    assert {c["id"] for c in series["conditions"] if c.get("image")} == {"B10", "B200", "B300", "P1500", "C70"}


def test_answers_are_not_in_the_series_file(series):
    a = json.loads((ROOT / "answers" / "S7.json").read_text(encoding="utf-8"))
    blob = json.dumps({k: v for k, v in series.items() if k != "_note"}, ensure_ascii=False)
    assert a["authors_conclusion"] not in blob
    for phrase in ("최적", "optim", "선택", "selected", "더 나아지지", "대전", "언더컷", "종횡비 20"):
        assert phrase not in blob, f"시리즈 파일(모델 입력)에 판단이 섞였다: {phrase}"


# ── 새 모듈 ─────────────────────────────────────────────────────────────
def test_depth_loss_is_lower_is_worse_with_tolerance():
    s = {"series_id": "T", "conditions": [{"id": "A", "etch_depth_um": 30.0},
                                          {"id": "B", "etch_depth_um": 29.0},
                                          {"id": "C", "etch_depth_um": 24.0}]}
    out = DepthLoss().rule(s, {k: {"ok": True} for k in "ABC"})
    rows = {r["condition"]: r for r in out["rows"]}
    assert rows["B"]["change_vs_first"] == "변화 없음"       # −3.3% 는 5% 안
    assert rows["C"]["change_vs_first"] == "악화" and rows["C"]["change_vs_first_pct"] == -20.0
    assert (out["worst_condition"], out["best_condition"]) == ("C", "A")


def test_depth_loss_is_not_judged_without_depth(series):
    j = checker.check(series, base_dir=str(ROOT / "data"))["judgeable"]["etch_stop"]
    assert not any(v["ok"] for v in j.values())     # 원문에 식각 중단 여부가 없다 — 지어내지 않는다


# ── 루브릭 ─────────────────────────────────────────────────────────────
def _grade(tmp_path, window, onset="B10", issue="taper"):
    run = {"run_id": "t", "pipeline_version": "v2", "series": "S7", "provider": "mock", "model": None,
           "mode": "single", "rag": False, "images": False,
           "single": {"q1": {"primary_issue": issue, "onset_condition": onset, "process_window": window},
                      "q2": {"hypotheses": []},
                      "q3": {"variable": "식각 단계 바이어스", "direction": "네 최적값을 조합해 사이클을 늘린다"}}}
    (tmp_path / "r.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    out_csv = tmp_path / "o.csv"
    subprocess.run([sys.executable, "-m", "harc.judge", str(tmp_path), "--csv", str(out_csv)],
                   capture_output=True, text=True, cwd=str(ROOT), check=True)
    rows = list(csv.DictReader(_io.open(out_csv, encoding="utf-8-sig")))
    return {r["field"]: r["verdict"] for r in rows if r["verdict"]}


def test_rubric_accepts_the_authors_recipe(tmp_path):
    v = _grade(tmp_path, "바이어스 300 V, ICP 1500 W, C4F8 70 sccm, SF6 120 sccm")
    assert v["q1.process_window"] == "correct"
    assert v["q1.primary_issue"] == "correct"
    assert v["q1.onset_condition"] == "correct"
    assert v["q3.direction"] == "correct"


def test_one_sweep_or_the_saturated_neighbour_is_half(tmp_path):
    for half in ("P1500", "B400 · P1700 · C80 · S130"):
        assert _grade(tmp_path, half)["q1.process_window"] == "acceptable", half


def test_the_standard_side_is_wrong(tmp_path):
    for bad in ("B10", "B200", "B300 · S150"):
        assert _grade(tmp_path, bad)["q1.process_window"] == "incorrect", bad


def test_label_matching_keeps_numeric_boundaries():
    from harc.judge import parse_conditions
    a = json.loads((ROOT / "answers" / "S7.json").read_text(encoding="utf-8"))
    assert parse_conditions("B10", a["conditions"]) == {"B10"}
    assert parse_conditions("바이어스 100 V", a["conditions"]) == {"B100"}
    assert parse_conditions("SF6 150 sccm", a["conditions"]) == {"S150"}
