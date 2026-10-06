"""S8 (kwon2024nano) — SiO2 200 nm 트렌치, CF4/H2/Ar, 척 온도 26 → −63 °C 스윕.

S8 을 넣은 이유: **목표 공정(SiO2 HARC)에 가장 가깝고, 저자 최적 조건이 한 지표에서는 최악이다.**
−63 °C 는 네킹·보잉·식각 깊이가 가장 좋지만 마스크 식각률이 가장 높다. 선택비는 오히려 오른다.
값은 래스터 그래프에서 읽었다 — 원문 정의식으로 서로 검증한다.
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
    return json.loads((ROOT / "data" / "series" / "S8.json").read_text(encoding="utf-8"))


def _rules(series):
    j = checker.check(series, base_dir=str(ROOT / "data"))["judgeable"]
    return {d.name: d.rule(series, j[d.name]) for d in ALL}


def test_figure_reads_agree_with_the_papers_definitions(series):
    """서로 다른 그림에서 읽은 값이 원문 식 (1)–(6) 으로 맞물려야 한다 — 판독 검증."""
    for c in series["conditions"]:
        # 식 (2): 마스크 식각률 = (1400 − 유효 마스크 두께)/6 min  (그림 7 의 두 곡선)
        assert (1400 - c["effective_mask_nm"]) / 6 == pytest.approx(c["mask_etch_rate_nm_min"], abs=1.0), c["id"]
        # 식 (1)·(3): 깊이 = 식각률 × 6, top = 깊이/종횡비  (그림 5)
        assert c["etch_depth_um"] * 1000 == pytest.approx(c["etch_rate_nm_min"] * 6, abs=5), c["id"]
        # 식 (5): 넥 CD = top × (1 − 네킹비)  (그림 8 의 두 곡선, 그림 5 와 교차)
        assert c["top_cd"] * (1 - c["necking_ratio"]) == pytest.approx(c["neck_cd"], abs=3), c["id"]
        # 식 (6): max_cd = top × (1 + 보잉비)  (그림 10)
        assert c["max_cd"] / c["top_cd"] - 1 == pytest.approx(c["bowing_ratio"], abs=0.01), c["id"]


def test_selectivity_matches_table_1_in_direction(series):
    """표 1(15분 식각)의 선택비 1.5 → 2.2 와 6분 식각 값이 같은 쪽으로 움직여야 한다."""
    c = {x["id"]: x for x in series["conditions"]}
    s = {k: v["etch_rate_nm_min"] / v["mask_etch_rate_nm_min"] for k, v in c.items()}
    assert s["T1"] == pytest.approx(1.5, abs=0.15) and s["T4"] == pytest.approx(2.2, abs=0.15)


def test_the_trap_exists(series):
    """저자 최적 T4 가 마스크 침식에서는 최악이고, 나머지 판정 가능한 지표에서는 최선이다."""
    r = _rules(series)
    assert r["mask_erosion"]["worst_condition"] == "T4"
    for name in ("bowing", "necking", "depth_loss", "cd_deviation"):
        assert r[name]["best_condition"] == "T4", name
    assert r["taper"]["worst_condition"] is None      # 하단 CD 가 원문에 없다 — 지어내지 않는다


def test_design_is_a_sweep(series):
    assert design_of(series) == "sweep" and show_trend(series) is True
    assert [c["chuck_temp_c"] for c in series["conditions"]] == [26, -22, -48, -63]


def test_every_condition_has_an_sem_image(series):
    for c in series["conditions"]:
        assert (ROOT / "data" / c["image"]).exists(), c["image"]


def test_answers_are_not_in_the_series_file(series):
    a = json.loads((ROOT / "answers" / "S8.json").read_text(encoding="utf-8"))
    blob = json.dumps({k: v for k, v in series.items() if k != "_note"}, ensure_ascii=False)
    assert a["authors_conclusion"] not in blob
    for phrase in ("최적", "optim", "점착", "sticking", "트위스팅", "twist", "방향성", "낮을수록", "reduction"):
        assert phrase not in blob, f"시리즈 파일(모델 입력)에 판단이 섞였다: {phrase}"


# ── 루브릭 ─────────────────────────────────────────────────────────────
def _grade(tmp_path, window, onset="T1 ~ T2", issue="necking", direction="척 온도를 −63 °C 이하로 더 낮춘다"):
    run = {"run_id": "t", "pipeline_version": "v2", "series": "S8", "provider": "mock", "model": None,
           "mode": "single", "rag": False, "images": False,
           "single": {"q1": {"primary_issue": issue, "onset_condition": onset, "process_window": window},
                      "q2": {"hypotheses": []},
                      "q3": {"variable": "척 온도", "direction": direction}}}
    (tmp_path / "r.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    out_csv = tmp_path / "o.csv"
    subprocess.run([sys.executable, "-m", "harc.judge", str(tmp_path), "--csv", str(out_csv)],
                   capture_output=True, text=True, cwd=str(ROOT), check=True)
    rows = list(csv.DictReader(_io.open(out_csv, encoding="utf-8-sig")))
    return {r["field"]: r["verdict"] for r in rows if r["verdict"]}


def test_rubric_accepts_the_authors_pick(tmp_path):
    v = _grade(tmp_path, "T4 (−63 ℃)")
    assert v["q1.process_window"] == "correct"
    assert v["q1.primary_issue"] == "correct"
    assert v["q1.onset_condition"] == "correct"
    assert v["q3.direction"] == "correct"


def test_the_mask_erosion_trap_is_wrong(tmp_path):
    """마스크 식각률만 보고 상온 쪽을 고르면 오답."""
    for bad in ("T1", "26 °C", "T1 ~ T2"):
        assert _grade(tmp_path, bad)["q1.process_window"] == "incorrect", bad
    assert _grade(tmp_path, "T4", direction="마스크 침식을 줄이려고 온도를 올린다")["q3.direction"] == "incorrect"


def test_negative_temperatures_are_read_in_any_notation():
    """온도 스윕에서 처음 드러난 것: 유니코드 마이너스·℃ 표기 (2026-09-30)."""
    from harc.judge import parse_conditions
    a = json.loads((ROOT / "answers" / "S8.json").read_text(encoding="utf-8"))["conditions"]
    for text in ("-63 °C", "−63 °C", "−63℃", "-63℃", "척 온도 -63 °C"):
        assert parse_conditions(text, a) == {"T4"}, text
    assert parse_conditions("-48 ~ -63 °C", a) == {"T3", "T4"}
    assert parse_conditions("26 °C", a) == {"T1"}
    assert parse_conditions("-630 °C", a) == set() and parse_conditions("126 °C", a) == set()
