"""S5 (you2025) — 순환 식각의 단계 시간, 두 OFAT 스윕이 중심 조건을 공유한다.

S5 를 넣은 이유: **반대 방향 결함의 트레이드오프**가 두 축에서 따로 나타난다. 증착·식각 단계가
짧으면 보잉, 길면 좁아짐이고 최적은 두 스윕의 중심(증착 7 s · 식각 30 s)이다.
모든 치수가 저자가 SEM 사진에 직접 적은 값이라 추출 오차가 없다. 그것을 고정한다.
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
from harc.defects.registry import ALL

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def series():
    return json.loads((ROOT / "data" / "series" / "S5.json").read_text(encoding="utf-8"))


@pytest.fixture
def answers():
    return json.loads((ROOT / "answers" / "S5.json").read_text(encoding="utf-8"))


def _by_id(series):
    return {c["id"]: c for c in series["conditions"]}


def test_values_are_the_ones_annotated_on_the_sem_images(series):
    c = _by_id(series)
    assert (c["D3"]["top_cd"], c["D3"]["neck_cd"], c["D3"]["max_cd"], c["D3"]["bottom_cd"]) == (197, 182, 236, 61)
    assert (c["D7"]["top_cd"], c["D7"]["max_cd"], c["D7"]["bottom_cd"]) == (196, 197, 153)
    assert (c["E20"]["top_cd"], c["E20"]["max_cd"], c["E20"]["bottom_cd"]) == (180, 229, 86)
    assert (c["E60"]["top_cd"], c["E60"]["bottom_cd"]) == (203, 81)


def test_cross_checks_against_the_text(series):
    """하단/상단은 원문 개구율과, (max-top)/2 는 원문 '보잉 정도' 와 맞아야 한다."""
    c = _by_id(series)
    opening = {"D3": 0.31, "D7": 0.78, "D9": 0.56, "E20": 0.48, "E40": 0.58, "E60": 0.40}
    for cid, want in opening.items():
        assert c[cid]["bottom_cd"] / c[cid]["top_cd"] == pytest.approx(want, abs=0.01), cid
    bow = {"D3": 19.5, "D5": 5, "D7": 0.5, "E20": 24.5}
    for cid, want in bow.items():
        assert (c[cid]["max_cd"] - c[cid]["top_cd"]) / 2 == pytest.approx(want), cid


def test_no_values_were_invented_for_narrowing_conditions(series):
    """원문이 최대 직경을 적지 않은 조건에 max_cd 를 만들어 넣으면 안 된다."""
    c = _by_id(series)
    for cid in ("D9", "E40", "E60"):
        assert "max_cd" not in c[cid]
    assert [k for k, v in c.items() if "neck_cd" in v] == ["D3"]


def test_design_is_ofat_so_the_overall_trend_is_off(series):
    assert design_of(series) == "ofat"
    assert show_trend(series) is False
    j = checker.check(series, base_dir=str(ROOT / "data"))["judgeable"]
    for d in ALL:
        out = d.rule(series, j[d.name])
        assert not [k for k in out if k.startswith("trend")], f"{d.name} 이 OFAT 에서 전체 추세를 냈다"


def test_rules_find_the_papers_optimum(series):
    j = checker.check(series, base_dir=str(ROOT / "data"))["judgeable"]
    rules = {d.name: d.rule(series, j[d.name]) for d in ALL}
    assert rules["bowing"]["best_condition"] == "D7" and rules["bowing"]["worst_condition"] == "E20"
    assert rules["taper"]["best_condition"] == "D7" and rules["taper"]["worst_condition"] == "D3"
    assert all(v["ok"] for v in j["etch_stop"].values())
    assert not any(v["ok"] for v in j["undercut"].values())


def test_every_condition_has_an_sem_image(series):
    have = [(ROOT / "data" / c["image"]).exists() for c in series["conditions"]]
    if not any(have):
        pytest.skip("공개본: 원 논문이 CC BY-NC-ND 라 SEM 크롭을 싣지 않았다 (tools/make_public.py)")
    for c, ok in zip(series["conditions"], have):
        assert ok, c["image"]           # 일부만 빠졌으면 실수다


def test_answers_are_not_in_the_series_file(series, answers):
    blob = json.dumps({k: v for k, v in series.items() if k != "_note"}, ensure_ascii=False)
    assert answers["authors_conclusion"] not in blob
    for key in ("authors_conclusion", "correct", "incorrect", "acceptable", "q1", "q2", "q3"):
        assert f'"{key}"' not in blob
    for phrase in ("최적", "optim", "공정 창", "process window", "보잉", "좁아짐", "narrowing", "bowing"):
        assert phrase not in blob, f"시리즈 파일(모델 입력)에 판단이 섞였다: {phrase}"


# ── 루브릭이 세 단계로 갈리는지 ──────────────────────────────────────────
def _grade(tmp_path, window, onset="D3", issue="bowing"):
    run = {"run_id": "t", "pipeline_version": "v2", "series": "S5", "provider": "mock", "model": None,
           "mode": "single", "rag": False, "images": False,
           "single": {"q1": {"primary_issue": issue, "onset_condition": onset, "process_window": window},
                      "q2": {"hypotheses": []},
                      "q3": {"variable": "증착 단계 시간", "direction": "증착 7 s · 식각 30 s 주변을 촘촘히"}}}
    (tmp_path / "r.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    out_csv = tmp_path / "o.csv"
    subprocess.run([sys.executable, "-m", "harc.judge", str(tmp_path), "--csv", str(out_csv)],
                   capture_output=True, text=True, cwd=str(ROOT), check=True)
    rows = list(csv.DictReader(_io.open(out_csv, encoding="utf-8-sig")))
    return {r["field"]: r["verdict"] for r in rows if r["verdict"]}


def test_rubric_accepts_the_authors_optimum(tmp_path):
    v = _grade(tmp_path, "D7 (증착 7 s / 식각 30 s)")
    assert v["q1.process_window"] == "correct"
    assert v["q1.primary_issue"] == "correct"
    assert v["q3.direction"] == "correct"


def test_onset_both_bowing_ends_is_correct_one_end_is_half(tmp_path):
    assert _grade(tmp_path, "D7", onset="D3, E20")["q1.onset_condition"] == "correct"
    assert _grade(tmp_path, "D7", onset="D3")["q1.onset_condition"] == "acceptable"


def test_rubric_gives_half_credit_to_a_wider_window(tmp_path):
    assert _grade(tmp_path, "D5 ~ D7")["q1.process_window"] == "acceptable"


def test_rubric_rejects_a_defect_condition_as_the_window(tmp_path):
    for bad in ("D3", "E20", "E60"):
        assert _grade(tmp_path, bad)["q1.process_window"] == "incorrect", bad


def test_naming_the_optimum_as_defect_onset_is_wrong(tmp_path):
    assert _grade(tmp_path, "D7", onset="D7")["q1.onset_condition"] == "incorrect"
