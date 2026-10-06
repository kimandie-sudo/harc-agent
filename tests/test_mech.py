"""기제 실험 v2 — 심은 청크 · 익명화 · 실행기 · 분석기 (DESIGN_기제실험.md). API 키 없이 돈다.

실험이 기대는 조건을 고정한다:
  ① 심은 청크는 최하위 **한 자리만** 바꾸고, 개수·위치·원문을 지킨다 (길이가 rag 와 같아야 한다)
  ② 표적은 **틀린** 조건이고, 표적 파일은 채점용 정답 파일과 분리돼 있다
  ③ 익명화하면 프롬프트에 실제 논문 id 가 하나도 없고, 인용은 되돌려 채점된다
  ④ 실행기는 재현 가능한 순서로 돌고, 재개되고, 연속 실패에 멈춘다
  ⑤ 분석기는 미완 블록을 버리고, 판정 규칙이 설계 문서 그대로다
"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from harc import rag, pipeline, cite_eval  # noqa: E402
import mech_run  # noqa: E402
import mech_analyze  # noqa: E402


def _series(s):
    return json.loads((ROOT / "data" / "series" / f"{s}.json").read_text(encoding="utf-8"))


def _answers(s):
    return json.loads((ROOT / "answers" / f"{s}.json").read_text(encoding="utf-8"))


# ── ② 표적 ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("sid", ["S1", "S2", "S3", "S4"])
def test_planted_target_is_a_wrong_condition_of_the_series(sid):
    spec = rag.planted_spec(_series(sid))
    pw = _answers(sid)["q1"]["process_window"]
    assert spec["target"] in {c["id"] for c in _series(sid)["conditions"]}
    assert spec["target"] not in pw.get("correct", []), "표적이 정답이면 양성 대조군이 아니다"
    assert spec["target"] not in pw.get("acceptable", []), "표적이 허용답이면 채택을 오답으로 못 센다"
    assert spec["target"] in spec["sentence"]


def test_planted_targets_live_outside_the_answer_key():
    """프롬프트 경로(rag.py)는 표적을 data/planted.json 에서 읽는다 — 정답 파일이 아니다."""
    assert rag.PLANTED_TARGETS == ROOT / "data" / "planted.json"
    assert rag.PLANTED_TARGETS.exists()


# ── ① 심은 청크 ─────────────────────────────────────────────────────────────
def _hits(n=6):
    return [{"chunk_id": f"p{i}#0", "paper_id": f"p{i}", "text": f"original text {i}"} for i in range(n)]


def test_planted_changes_only_the_last_chunk_and_keeps_its_text():
    hits = _hits()
    out = rag.planted_for(hits, _series("S2"))
    assert len(out) == len(hits)
    assert out[:-1] == hits[:-1], "최하위 말고는 건드리면 안 된다"
    spec = rag.planted_spec(_series("S2"))
    assert out[-1]["text"] == spec["sentence"] + " " + hits[-1]["text"], "원문을 지우지 않고 앞에 붙인다"
    assert out[-1]["chunk_id"] == rag.PLANTED_MARK + hits[-1]["chunk_id"]
    assert hits[-1]["chunk_id"] == "p5#0", "입력 목록을 고치면 안 된다"


def test_planted_on_empty_hits_is_empty():
    assert rag.planted_for([], _series("S1")) == []


def test_planted_requires_anonymization():
    """익명화 없이 쓰면 'planted:' 표지가 프롬프트에 그대로 드러난다."""
    with pytest.raises(ValueError, match="anonymize"):
        pipeline.run(_series("S1"), provider="mock", mode="single", use_rag=True,
                     base_dir=str(ROOT / "data"), chunk_source="planted", anonymize_chunks=False)


# ── ③ 익명화 ───────────────────────────────────────────────────────────────
@pytest.fixture
def captured(monkeypatch):
    seen = []
    real = pipeline.call_json

    def spy(agent, **kw):
        seen.append(kw.get("user", ""))
        return real(agent, **kw)
    monkeypatch.setattr(pipeline, "call_json", spy)
    return seen


@pytest.mark.parametrize("source", ["rag", "placebo", "planted"])
def test_anonymized_prompt_has_no_real_chunk_ids(captured, source):
    if source == "placebo" and not rag.PLACEBO_CHUNKS.exists():
        pytest.skip("위약 색인 없음")
    if not rag.CHUNKS.exists():
        pytest.skip("검색 색인 없음 (공개본은 저작권 때문에 코퍼스를 싣지 않는다)")
    out = pipeline.run(_series("S2"), provider="mock", mode="single", use_rag=True,
                       base_dir=str(ROOT / "data"), chunk_source=source, anonymize_chunks=True)
    prompt = captured[-1]
    real = [c["chunk_id"] for c in out["rag_chunks"]]
    assert real, "청크가 없으면 검사가 무의미하다"
    assert out["chunk_alias"] == {f"D{i + 1}": cid for i, cid in enumerate(real)}
    for i in range(len(real)):
        assert f'"D{i + 1}"' in prompt
    leaked = [c for c in real if (c.replace(rag.PLANTED_MARK, "") in prompt or c.replace(rag.PLANTED_MARK, "").split("#")[0] in prompt)]
    assert not leaked, f"프롬프트에 실제 id 가 샜다: {leaked}"
    assert "planted" not in prompt
    if source == "planted":
        assert out["planted"]["sentence"] in prompt
        assert out["rag_chunks"][-1]["chunk_id"].startswith(rag.PLANTED_MARK)


def test_rag_off_has_no_alias():
    out = pipeline.run(_series("S1"), provider="mock", mode="single", use_rag=False,
                       base_dir=str(ROOT / "data"), anonymize_chunks=True)
    assert "chunk_alias" not in out and out["rag_chunks"] == []


def test_cite_eval_resolves_aliases_and_still_catches_hallucination():
    run = {"mode": "single", "rag": True, "run_id": "t",
           "rag_chunks": [{"chunk_id": "a#1", "text": "bowing passivation"},
                          {"chunk_id": "b#2", "text": "taper ion"}],
           "chunk_alias": {"D1": "a#1", "D2": "b#2"},
           "single": {"q2": {"hypotheses": [{"mechanism": "ion taper", "citations": ["D2", "D9"]}]}}}
    rows = {r["chunk_id"]: r for r in cite_eval.citation_rows(run, vocab={"ion", "taper"})}
    assert rows["b#2"]["in_retrieved"] is True and rows["b#2"]["rank"] == 2
    assert rows["D9"]["in_retrieved"] is False, "대응표에 없는 인용은 환각으로 남아야 한다"
    assert cite_eval.uncited_ranks(run) == [1]


# ── ④ 실행기 ───────────────────────────────────────────────────────────────
def test_schedule_is_deterministic_and_rep_major():
    a, b = mech_run.schedule(), mech_run.schedule()
    assert a == b
    assert len(a) == 4 * 2 * 10
    assert all(sorted(x["arms"]) == sorted(mech_run.ARMS) for x in a)
    assert [x["rep"] for x in a[:8]] == [1] * 8, "반복이 바깥이어야 끊겨도 셀이 고르게 남는다"
    assert len({tuple(x["arms"]) for x in a}) > 10, "팔 순서가 블록마다 섞여야 한다"
    assert mech_run.schedule(seed=1) != a


def test_every_arm_is_single_mode_bm25_and_anonymized():
    assert mech_run.COMMON["mode"] == "single"
    assert mech_run.COMMON["retriever"] == "bm25"
    assert mech_run.COMMON["anonymize_chunks"] is True
    assert mech_run.COMMON["temperature"] == 0.0
    assert mech_run.ARMS["off"] == mech_run.ARMS["off2"], "off′ 는 off 와 입력이 같아야 잡음 바닥이다"


def _fake_runner(fail_on=()):
    calls = []

    def run(series, model, kw):
        calls.append((series["series_id"], model, kw.get("chunk_source"), kw["use_rag"]))
        if len(calls) in fail_on:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return {"run_id": f"r{len(calls)}", "series": series["series_id"], "mode": "single"}
    run.calls = calls
    return run


def test_run_schedule_resumes(tmp_path):
    sched = mech_run.schedule(series=("S1",), models=("m",), reps=2)
    r1 = _fake_runner()
    s1 = mech_run.run_schedule(sched, tmp_path, r1, log=lambda *_: None)
    assert s1["done"] == 10 and len(list(tmp_path.glob("*.json"))) == 10
    r2 = _fake_runner()
    s2 = mech_run.run_schedule(sched, tmp_path, r2, log=lambda *_: None)
    assert s2["skipped"] == 10 and r2.calls == []
    one = json.loads(next(tmp_path.glob("*__planted.json")).read_text(encoding="utf-8"))
    assert one["mech"]["arm"] == "planted" and one["mech"]["design"] == "v2"


def test_run_schedule_stops_after_three_consecutive_failures(tmp_path):
    sched = mech_run.schedule(series=("S1",), models=("m",), reps=2)
    r = _fake_runner(fail_on=(2, 3, 4))
    s = mech_run.run_schedule(sched, tmp_path, r, log=lambda *_: None)
    assert s["stopped"] and s["failed"] == 3 and s["done"] == 1
    assert len(r.calls) == 4, "세 번째 연속 실패에서 바로 멈춰야 한다"


def test_run_schedule_resets_the_streak_on_success(tmp_path):
    sched = mech_run.schedule(series=("S1",), models=("m",), reps=2)
    r = _fake_runner(fail_on=(1, 2, 4, 5))
    s = mech_run.run_schedule(sched, tmp_path, r, log=lambda *_: None)
    assert not s["stopped"] and s["failed"] == 4


def test_schedule_file_refuses_a_different_design(tmp_path):
    d1 = mech_run.design_of(("S1",), ("m",), 2, 1)
    mech_run.write_or_check_schedule(tmp_path, d1, [])
    mech_run.write_or_check_schedule(tmp_path, d1, [])          # 같은 설계면 통과
    with pytest.raises(SystemExit):
        mech_run.write_or_check_schedule(tmp_path, mech_run.design_of(("S1",), ("m",), 3, 1), [])


# ── ⑤ 분석기 ───────────────────────────────────────────────────────────────
def _run(block, arm, series, pw):
    return {"run_id": f"{block}-{arm}", "series": series, "mode": "single", "rag_chunks": [],
            "single": {"q1": {"primary_issue": "taper", "onset_condition": "L1", "process_window": pw}},
            "mech": {"block": block, "arm": arm, "design": "v2"}}


def _write(tmp_path, runs):
    for r in runs:
        (tmp_path / f"{r['mech']['block']}__{r['mech']['arm']}.json").write_text(
            json.dumps(r, ensure_ascii=False), encoding="utf-8")


def test_analyzer_drops_incomplete_blocks(tmp_path):
    runs = [_run("b1", a, "S2", "L4") for a in mech_analyze.ARMS]
    runs += [_run("b2", a, "S2", "L4") for a in ("off", "rag")]
    _write(tmp_path, runs)
    res = mech_analyze.analyze(tmp_path)
    assert res["blocks"] == 1 and res["dropped_incomplete"] == 1


def test_analyzer_detects_adoption_and_movement(tmp_path):
    """planted 가 표적(L2)을 늘 따르고 rag 는 L3 로 움직이고 placebo 는 가만히 있는 세계 → C."""
    runs = []
    for i in range(30):
        b = f"b{i}"
        runs += [_run(b, "off", "S2", "L4"), _run(b, "off2", "S2", "L4"),
                 _run(b, "rag", "S2", "L3"), _run(b, "placebo", "S2", "L4"),
                 _run(b, "planted", "S2", "L2")]
    _write(tmp_path, runs)
    res = mech_analyze.analyze(tmp_path)
    assert res["adoption_rate"]["planted"] == 1.0 and res["adoption_rate"]["off2"] == 0.0
    assert res["movement_rate"]["rag"] == 1.0 and res["movement_rate"]["placebo"] == 0.0
    assert res["primary"]["P1"]["p"] < mech_analyze.ALPHA
    assert res["verdict"] == "C"


def test_range_answer_that_includes_the_correct_condition_is_not_adoption():
    """'L1 ~ L4' 는 L2 를 포함하지만 정답 L4 도 포함한다 — 채택이 아니다."""
    assert mech_analyze.adopted(_run("b", "planted", "S2", "L1 ~ L4"), "L2") == 0
    assert mech_analyze.adopted(_run("b", "planted", "S2", "L2"), "L2") == 1


@pytest.mark.parametrize("p,expect", [
    ((0.5, 0.001, 0.001), "0"),        # P1 실패 → 나머지와 무관하게 멈춘다
    ((0.5, 0.001, 0.001, 0.01), "0R"), # P1 실패인데 심은 청크를 인용은 한다 → 규칙이 이긴다
    ((0.001, 0.5, 0.5), "A"),
    ((0.001, 0.001, 0.5), "C"),
    ((0.001, 0.001, 0.001), "B"),
    ((0.001, 0.5, 0.001), "?"),
])
def test_decision_rule_matches_the_design(p, expect):
    assert mech_analyze.decide(*p)[0] == expect


def test_sign_flip_test_is_null_on_zero_differences():
    assert mech_analyze.sign_flip_test([0] * 40) == (0.0, 1.0)
    d, p = mech_analyze.sign_flip_test([1] * 10, "greater")
    assert d == 1.0 and p < 0.01


def test_intrusion_counts_only_tokens_from_chunks_not_in_the_series():
    run = _run("b", "rag", "S2", "L4")
    run["rag_chunks"] = [{"chunk_id": "x", "text": "at 10 mTorr and 500 W"}]
    run["single"]["q1"]["tradeoffs"] = ["10 mTorr 에서 개선"]
    assert mech_analyze.intrusion(run) == 1
    run["single"]["q1"]["tradeoffs"] = ["차이 없음"]
    assert mech_analyze.intrusion(run) == 0


def test_reading_indicator_compares_the_same_slot(tmp_path):
    """P1 실패 + 심은 자리(D6)를 rag 때보다 더 인용 → 0R (읽지만 규칙을 따른다)."""
    runs = []
    for i in range(30):
        b = f"b{i}"
        for arm in mech_analyze.ARMS:
            r = _run(b, arm, "S2", "L4")                      # 공정 창은 아무도 안 바뀐다
            if arm in ("rag", "placebo", "planted"):
                r["chunk_alias"] = {f"D{k}": f"x#{k}" for k in range(1, 7)}
                cites = ["D6"] if arm == "planted" else ["D1"]
                r["single"]["q2"] = {"hypotheses": [{"mechanism": "m", "citations": cites}]}
            runs.append(r)
    _write(tmp_path, runs)
    res = mech_analyze.analyze(tmp_path)
    assert res["reading"]["cited_last_rate"] == {"rag": 0.0, "planted": 1.0}
    assert res["verdict"] == "0R"


@pytest.mark.parametrize("q2", ["(문자열)", None, {"hypotheses": "문자열"}, {"hypotheses": [{"citations": "D6"}]}])
def test_cited_last_survives_off_schema_answers(q2):
    """mock 은 q2 를 문자열로 준다. 실제 모델도 스키마를 벗어날 수 있다 — 분석이 죽으면 안 된다."""
    r = _run("b", "planted", "S2", "L4")
    r["chunk_alias"] = {f"D{k}": f"x#{k}" for k in range(1, 7)}
    r["single"]["q2"] = q2
    assert mech_analyze.cited_last(r) == (1 if isinstance(q2, dict) and q2.get("hypotheses") != "문자열" else 0)
