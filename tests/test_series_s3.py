"""S3 (park2025psst) 시리즈와 necking 결함 모듈.

S3 는 원문이 조건별 절대 CD 를 싣지 않아 **보잉·테이퍼가 판정 불가**인 시리즈다.
그 상태가 의도대로 유지되는지, 그리고 루브릭이 정답과 오답을 **양방향으로** 가르는지
고정한다. 루브릭은 한쪽만 검사하면 영영 0점인 채로 통과할 수 있다.
"""
import json
import pathlib

import pytest

from harc import checker
from harc.defects.registry import ALL
from harc.defects.necking import Necking

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def series():
    return json.loads((ROOT / "data" / "series" / "S3.json").read_text(encoding="utf-8"))


@pytest.fixture
def answers():
    return json.loads((ROOT / "answers" / "S3.json").read_text(encoding="utf-8"))


def test_series_matches_the_paper(series):
    """원문 수치 — 기준 대비 넥 CD +14%/-33%, 하단 CD +32%/-29%."""
    c = {x["id"]: x for x in series["conditions"]}
    assert set(c) == {"R", "P90", "P180"}
    assert (c["R"]["neck_cd"], c["R"]["bottom_cd"]) == (100, 100)
    assert (c["P90"]["neck_cd"], c["P90"]["bottom_cd"]) == (114, 132)
    assert (c["P180"]["neck_cd"], c["P180"]["bottom_cd"]) == (67, 71)
    assert all(not x["etch_stop"] for x in series["conditions"]), \
        "P180 의 실패 양상은 식각 중단이 아니라 과식각(punch-through)이다"


def test_answers_are_not_in_the_series_file(series, answers):
    """정답은 answers/ 에만. 시리즈 파일에 새어 들어가면 과제가 무너진다.

    조건 ID(P90 등)는 두 파일에 모두 있어야 한다 — 그건 누수가 아니다.
    새면 안 되는 것은 **어느 조건이 정답인지**와 저자 결론·근거 문장이다.
    """
    blob = json.dumps(series, ensure_ascii=False)
    assert answers["authors_conclusion"] not in blob
    for key in ("authors_conclusion", "q1", "q2", "q3", "correct", "incorrect", "acceptable"):
        assert f'"{key}"' not in blob, f"시리즈 파일에 루브릭 키: {key}"
    for q in ("primary_issue", "process_window", "onset_condition"):
        blk = answers["q1"][q]
        for txt in (blk.get("note"), blk.get("quote")):
            if txt:
                assert txt not in blob, f"시리즈 파일에 루브릭 해설: {q}"
    # 저자 결론의 특징적인 구절이 시리즈 파일에 없어야 한다
    for phrase in ("최선", "최적", "개선된다", "나빴다", "punched", "과식각"):
        assert phrase not in blob, f"시리즈 파일에 판단이 섞였다: {phrase}"


def test_bowing_and_taper_stay_unjudgeable(series):
    """원문에 top_cd·max_cd 가 없다. '판정 불가'가 올바른 상태이며 추측하면 안 된다."""
    res = checker.check(series, base_dir=str(ROOT / "data"))
    j = res["judgeable"]
    for name in ("bowing", "taper"):
        assert all(v["ok"] is False for v in j[name].values()), \
            f"{name} 이 판정 가능해졌다 — 데이터가 바뀌었는지 확인하라"
    for name in ("necking", "etch_stop"):
        assert all(v["ok"] for v in j[name].values()), f"{name} 이 판정 불가가 됐다"


def test_necking_reads_the_direction_correctly(series):
    """넥 CD 는 **클수록 좋다** — bowing·taper 와 방향이 반대다."""
    res = checker.check(series, base_dir=str(ROOT / "data"))
    out = Necking().rule(series, res["judgeable"]["necking"])
    assert out["direction"] == "lower_is_worse"
    assert out["best_condition"] == "P90" and out["worst_condition"] == "P180"
    verdict = {r["condition"]: r["change_vs_first"] for r in out["rows"]}
    assert verdict["P90"] == "개선" and verdict["P180"] == "악화"


def test_necking_is_registered():
    assert "necking" in {d.name for d in ALL}


def test_necking_without_top_cd_skips_the_ratio(series):
    """top_cd 가 없으면 넥/상단 비는 내지 않는다 — 없는 값을 지어내지 않는다."""
    res = checker.check(series, base_dir=str(ROOT / "data"))
    out = Necking().rule(series, res["judgeable"]["necking"])
    assert all("neck_ratio" not in r for r in out["rows"])


def test_necking_uses_the_ratio_when_top_cd_exists():
    s = {"series_id": "T", "conditions": [
        {"id": "A", "neck_cd": 90, "top_cd": 100},      # 비 0.90 < 0.95 → 네킹
        {"id": "B", "neck_cd": 98, "top_cd": 100},      # 비 0.98 → 아님
    ]}
    judge = {c["id"]: {"ok": True} for c in s["conditions"]}
    rows = {r["condition"]: r for r in Necking().rule(s, judge)["rows"]}
    assert rows["A"]["neck_ratio"] == 0.9 and rows["A"]["necking"] is True
    assert rows["B"]["necking"] is False


# ── 루브릭이 양방향으로 동작하는지 ────────────────────────────────────────
def _grade(tmp_path, aggregate, cause, nxt):
    """가상의 런 하나를 채점해 {필드: 판정} 을 돌려준다."""
    import subprocess, sys, csv, io as _io
    run = {"run_id": "t", "pipeline_version": "v2", "series": "S3", "provider": "mock",
           "model": None, "mode": "multi", "rag": False, "images": False,
           "aggregate": aggregate, "cause": cause, "next": nxt}
    (tmp_path / "r.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    csv_path = tmp_path / "out.csv"
    subprocess.run([sys.executable, "-m", "harc.judge", str(tmp_path), "--csv", str(csv_path)],
                   capture_output=True, text=True, cwd=str(ROOT), check=True)
    rows = list(csv.DictReader(_io.open(csv_path, encoding="utf-8-sig")))
    return {r["field"]: r["verdict"] for r in rows if r["verdict"]}


GOLD_AGG = {"primary_issue": "necking (입구 좁아짐)",
            "process_window": "Δphase 90° (P90) 가 넥 CD 114, 하단 CD 132 로 최선",
            "ranking": "", "tradeoffs": ""}
GOLD_NEXT = {"recommendation": "상·하부 전극의 60 MHz 소스 RF 전력 비를 최적화하고 "
                               "Δphase 90° 부근을 더 촘촘히 쪼개 본다",
             "alternatives": "", "measure_next": ""}
BAD_AGG = {"primary_issue": "bowing",
           "process_window": "Δphase 180° (P180) 이 최적",
           "ranking": "", "tradeoffs": ""}
BAD_NEXT = {"recommendation": "위상 제어를 끄고 기준 조건으로 되돌린다",
            "alternatives": "", "measure_next": ""}
CAUSE = {"hypotheses": "Δphase 가 시스 두께와 플라즈마 밀도를 바꾼다. 90°에서 상·하부 "
                       "Vrms 비가 최대가 되어 폴리머 패시베이션이 강화된다.",
         "cannot_assess": ""}


def test_rubric_accepts_the_authors_conclusion(tmp_path):
    v = _grade(tmp_path, GOLD_AGG, CAUSE, GOLD_NEXT)
    for f in ("q1.primary_issue", "q1.process_window", "q3.direction", "q3.variable"):
        assert v.get(f) == "correct", f"{f} 를 정답으로 인정하지 못한다: {v}"


def test_rubric_rejects_the_direction_flip(tmp_path):
    """최악(P180)을 최선으로 부르고 판정 불가인 bowing 을 주 문제로 꼽는 답."""
    v = _grade(tmp_path, BAD_AGG, CAUSE, BAD_NEXT)
    for f in ("q1.primary_issue", "q1.process_window", "q3.direction"):
        assert v.get(f) == "incorrect", f"{f} 를 오답으로 잡지 못한다: {v}"
