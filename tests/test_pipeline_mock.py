"""mock provider 로 파이프라인 전체를 돌린다 (API 키 없이 구조만 검증)."""
import pytest
from harc import PIPELINE_VERSION
from harc import pipeline
from harc.defects.registry import ALL


@pytest.fixture
def recorded(monkeypatch):
    """LLM 호출의 system/user 프롬프트를 모아 둔다."""
    calls = []
    real = pipeline.call_json
    def spy(agent, provider, system, user, *a, **kw):
        calls.append({"agent": agent, "system": system, "user": user})
        return real(agent, provider, system, user, *a, **kw)
    monkeypatch.setattr(pipeline, "call_json", spy)
    return calls


@pytest.mark.parametrize("sid", ["S1", "S2"])
@pytest.mark.parametrize("mode", ["multi", "single"])
def test_run_produces_v2_output(series, sid, mode):
    out = pipeline.run(series(sid), provider="mock", mode=mode, base_dir="data")
    assert out["pipeline_version"] == PIPELINE_VERSION == "v5.1"
    assert out["series"] == sid and out["mode"] == mode
    # 레지스트리에서 끌어온다 — 결함 모듈을 더해도 이 테스트가 깨지지 않게
    from harc.defects.registry import ALL
    names = {d.name for d in ALL}
    assert set(out["rules"]) == names
    assert "taper_narrowing" not in out["rules"]
    for r in out["rules"].values():
        assert r["direction"] and r["summary"]
        assert "worst_condition" in r and "best_condition" in r
    if mode == "multi":
        assert set(out["defects"]) == names
        for key in ("aggregate", "cause", "next"):
            assert out[key]
    else:
        assert set(out["single"]) == {"q1", "q2", "q3"}


def test_single_prompt_carries_every_defect_definition(series, recorded):
    pipeline.run(series("S2"), provider="mock", mode="single", base_dir="data")
    system = next(c["system"] for c in recorded if c["agent"] == "single")
    for d in ALL:
        assert d.name in system and d.definition in system
        assert d.direction in system and d.direction_note in system


def test_single_and_multi_state_the_same_directions(series, recorded):
    pipeline.run(series("S2"), provider="mock", mode="single", base_dir="data")
    single = next(c["system"] for c in recorded if c["agent"] == "single")
    recorded.clear()
    pipeline.run(series("S2"), provider="mock", mode="multi", base_dir="data")
    for d in ALL:
        multi = next(c["system"] for c in recorded if c["agent"] == f"defect:{d.name}")
        assert d.direction in multi and d.direction_note in multi
        assert d.direction_note in single      # single 도 같은 방향 설명을 받는다


def test_rag_warning_when_no_chunks(series, tmp_path, monkeypatch):
    from harc import rag
    monkeypatch.setattr(rag, "CHUNKS", tmp_path / "chunks.jsonl")
    monkeypatch.setattr(rag, "PDF_DIR", tmp_path / "pdfs")
    out = pipeline.run(series("S1"), provider="mock", mode="multi", use_rag=True, base_dir="data")
    assert out["rag_chunks"] == []
    assert any("청크가 0개" in w for w in out["rag_warning"])


def test_no_rag_warning_when_rag_off(series):
    out = pipeline.run(series("S1"), provider="mock", mode="single", base_dir="data")
    assert out["rag_warning"] == []


def test_stray_pdf_name_reaches_the_run_json(series, tmp_path, monkeypatch):
    from harc import rag
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"chunk_id": "x#0", "paper_id": "x", "text": "bowing taper sidewall passivation"}\n',
                      encoding="utf-8")
    pdfs = tmp_path / "pdfs"; pdfs.mkdir()
    (pdfs / "coatings_2023_v2.pdf").write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(rag, "CHUNKS", chunks)
    monkeypatch.setattr(rag, "PDF_DIR", pdfs)
    monkeypatch.setattr(rag, "_INDEX", None)
    out = pipeline.run(series("S1"), provider="mock", mode="single", use_rag=True, base_dir="data")
    assert any("coatings_2023_v2.pdf" in w for w in out["rag_warning"])


# --- cause 프롬프트의 인용 면책 조항 (--cause-hedge) ---
# outputs_rag/ 32런에서 multi(cause)는 rag-on 8런 모두 인용 0건, single 은 8런 모두 인용이었다.
# 두 프롬프트의 인용 지시가 달랐던 것이 유력한 원인이라 hedge 를 끌 수 있게 했다.

HEDGE_SENTENCE = '청크에 없는 내용은 citations 를 비우고 basis 를 "general_knowledge"로 표시하세요'


def test_hedged_cause_prompt_offers_the_escape_hatch():
    t = pipeline.cause_system(True)
    assert HEDGE_SENTENCE in t
    assert '"basis"' in t


def test_unhedged_cause_prompt_drops_hatch_and_basis_field():
    t = pipeline.cause_system(False)
    assert HEDGE_SENTENCE not in t
    assert '"basis"' not in t          # 스키마에서도 빠져야 '안 썼다'고 답할 자리가 없다
    assert "chunk_id 를 citations 에 넣으세요" in t   # 인용 지시 자체는 남는다


def test_hedge_default_is_the_v2_prompt():
    """기본값은 v2 까지의 판이어야 한다 — outputs_rag/ 32런의 대조군이 그대로 재현된다."""
    assert pipeline.cause_system() == pipeline.cause_system(True)


def test_cause_hedge_is_recorded_in_output(series):
    for hedge in (True, False):
        out = pipeline.run(series("S2"), provider="mock", mode="multi",
                           base_dir="data", cause_hedge=hedge)
        assert out["cause_hedge"] is hedge


def test_cause_hedge_reaches_the_cause_call_only(series, recorded):
    pipeline.run(series("S2"), provider="mock", mode="multi", use_rag=False,
                 base_dir="data", cause_hedge=False)
    cause = [c for c in recorded if c["agent"] == "cause"]
    assert len(cause) == 1
    assert HEDGE_SENTENCE not in cause[0]["system"]
    for c in recorded:                      # 다른 단계 프롬프트는 건드리지 않는다
        if c["agent"] != "cause":
            assert HEDGE_SENTENCE not in c["system"]


def test_oversized_chunk_is_warned_not_silently_cut(monkeypatch):
    """옛 판 색인을 쓰면 조용히 잘리지 않고 경고가 남아야 한다."""
    warnings = []
    hit = {"chunk_id": "x#0", "text": "가" * (pipeline.CHUNK_CHAR_CAP + 500)}
    out = pipeline._pass_chunk(hit, warnings)
    assert len(out["text"]) == pipeline.CHUNK_CHAR_CAP
    assert warnings and "옛 판" in warnings[0] and "x#0" in warnings[0]


def test_normal_chunk_passes_through_whole(monkeypatch):
    warnings = []
    hit = {"chunk_id": "x#0", "text": "가" * 2000}
    assert pipeline._pass_chunk(hit, warnings)["text"] == hit["text"]
    assert warnings == []


def test_chunk_chars_cuts_on_purpose_without_warning():
    """--chunk-chars 는 실험 조건이다. 일부러 자르는 것이므로 '옛 판 색인' 경고를 내면 안 된다."""
    warnings = []
    hit = {"chunk_id": "x#0", "text": "가" * 2000}
    out = pipeline._pass_chunk(hit, warnings, chunk_chars=1200)
    assert len(out["text"]) == 1200
    assert warnings == []


def test_stale_index_still_warns_when_no_cut_requested():
    warnings = []
    hit = {"chunk_id": "x#0", "text": "가" * (pipeline.CHUNK_CHAR_CAP + 100)}
    pipeline._pass_chunk(hit, warnings, chunk_chars=None)
    assert warnings and "옛 판" in warnings[0]


def test_chunk_chars_is_recorded_in_output(series):
    for cut in (None, 1200):
        out = pipeline.run(series("S2"), provider="mock", mode="multi", use_rag=True,
                           base_dir="data", chunk_chars=cut)
        assert out["chunk_chars"] == cut
        if cut:
            assert all(len(c["text"]) <= cut for c in out["rag_chunks"])


# --- temperature ---
# 기본 0.0 은 사실상 결정론이라 --reps 를 올려도 서버측 비결정성만 재게 된다.
# 반복으로 변동을 보려면 온도를 올려야 하므로, 그것이 **모든** 호출에 실려야 한다.

def test_temperature_reaches_every_call(series, recorded, monkeypatch):
    """한 군데라도 빠지면 그 단계만 결정론으로 남아 반복 실험이 조용히 망가진다."""
    seen = []
    real = pipeline.call_json
    def spy(agent, provider, system, user, *a, **kw):
        seen.append((agent, kw.get("temperature")))
        return real(agent, provider, system, user, *a, **kw)
    monkeypatch.setattr(pipeline, "call_json", spy)
    pipeline.run(series("S2"), provider="mock", mode="multi", base_dir="data", temperature=0.7)
    assert seen, "호출이 없었다"
    wrong = [(a, t) for a, t in seen if t != 0.7]
    assert not wrong, f"temperature 가 안 실린 호출: {wrong}"


def test_temperature_reaches_the_image_checker(series, monkeypatch):
    """checker 는 pipeline 과 별도 경로다 — 여기가 빠지기 쉽다."""
    from harc import checker
    seen = []
    real = checker.call_json
    def spy(agent, provider, system, user, *a, **kw):
        seen.append((agent, kw.get("temperature")))
        return real(agent, provider, system, user, *a, **kw)
    monkeypatch.setattr(checker, "call_json", spy)
    # S1 — 사진이 CC BY 라 공개본에도 있다 (S2 는 NC-ND 라 공개본에서 빠진다)
    pipeline.run(series("S1"), provider="mock", mode="single", use_images=True,
                 base_dir="data", temperature=0.9)
    assert ("checker_image", 0.9) in seen, seen


def test_temperature_defaults_to_zero_and_is_recorded(series):
    out = pipeline.run(series("S2"), provider="mock", mode="multi", base_dir="data")
    assert out["temperature"] == 0.0
    out = pipeline.run(series("S2"), provider="mock", mode="multi", base_dir="data",
                       temperature=0.7)
    assert out["temperature"] == 0.7


# --- 규칙 방향 힌트 제거 (--rule-hints off) ---
# 채점 항목(onset=worst_condition, process_window=best_condition)이 코드가 계산해
# 프롬프트에 넣어 준 값과 같아서, 모델은 받은 값을 옮기기만 해도 맞는다.
# 힌트를 빼야 문헌이 기여할 자리가 생긴다.

HINT_VALUES = ["higher_is_worse", "true_is_worse", "lower_is_worse", "클수록 심함", "작을수록 심함",
               "worst_condition", "best_condition", '"summary"', '"onset"', "direction_note"]


def _all_prompts(series, recorded, **kw):
    pipeline.run(series("S2"), provider="mock", base_dir="data", **kw)
    return [(c["agent"], (c["system"] or "") + "\n" + (c["user"] or "")) for c in recorded]


@pytest.mark.parametrize("mode", ["multi", "single"])
def test_no_direction_hint_survives_anywhere(series, recorded, mode):
    """한 군데라도 남으면 ablation 이 무의미하다. 프롬프트 전체를 본다."""
    leaks = [(a, h) for a, t in _all_prompts(series, recorded, mode=mode, use_rag=False,
                                             rule_hints=False)
             for h in HINT_VALUES if h in t]
    assert not leaks, f"방향 힌트가 남았다: {leaks}"


@pytest.mark.parametrize("mode", ["multi", "single"])
def test_hints_on_keeps_them(series, recorded, mode):
    """대조군은 v2 그대로여야 한다."""
    txt = "\n".join(t for _, t in _all_prompts(series, recorded, mode=mode, use_rag=False,
                                               rule_hints=True))
    assert "higher_is_worse" in txt and "worst_condition" in txt


def test_recorded_rules_stay_complete_either_way(series):
    """결과 JSON 의 rules 는 기록이자 채점 기준이므로 항상 온전해야 한다."""
    for hints in (True, False):
        out = pipeline.run(series("S2"), provider="mock", mode="multi", base_dir="data",
                           rule_hints=hints)
        assert out["rule_hints"] is hints
        r = out["rules"]["taper"]
        for k in ("direction", "worst_condition", "best_condition", "summary"):
            assert k in r, (hints, k)


def test_strip_direction_hints_keeps_the_numbers():
    """방향만 빼고 값은 남겨야 한다 — 뺀 게 아니라 지운 것이면 비교가 불공정해진다."""
    rules = {"taper": {"metric": "top_bottom_ratio", "direction": "higher_is_worse",
                       "worst_condition": "L1", "summary": "…",
                       "rows": [{"condition": "L1", "top_bottom_ratio": 4.0,
                                 "change_vs_first": "변화 없음", "change_vs_first_pct": 0.0}],
                       "factorial": {"main_effects_pct": {"gas": [{"change_pct": -25.0,
                                                                   "verdict": "개선"}]},
                                     "direction": "higher_is_worse", "note": "…"}}}
    s = pipeline.strip_direction_hints(rules)["taper"]
    assert s["metric"] == "top_bottom_ratio"
    assert s["rows"][0]["top_bottom_ratio"] == 4.0          # 값은 남는다
    assert s["rows"][0]["change_vs_first_pct"] == 0.0       # 부호 있는 변화율도 값이다
    assert "change_vs_first" not in s["rows"][0]            # 개선/악화 판정은 뺀다
    for k in ("direction", "worst_condition", "summary"):
        assert k not in s
    assert "verdict" not in s["factorial"]["main_effects_pct"]["gas"][0]
    assert "direction" not in s["factorial"] and "note" not in s["factorial"]
    assert s["factorial"]["main_effects_pct"]["gas"][0]["change_pct"] == -25.0


def test_original_rules_are_not_mutated():
    rules = {"taper": {"direction": "higher_is_worse", "rows": [{"change_vs_first": "개선"}]}}
    pipeline.strip_direction_hints(rules)
    assert rules["taper"]["direction"] == "higher_is_worse"
    assert rules["taper"]["rows"][0]["change_vs_first"] == "개선"


def test_required_keys_reject_blank_values():
    """키가 있어야 하고, nullable 이 아닌 한 값도 비면 안 된다.

    예전 판은 `k not in obj` 만 봐서 모델이 키를 빠뜨려도 값을 비워도 통과했다.
    """
    import pytest
    from harc.llm import _check_required, _is_blank

    for blank in (None, "", "   "):
        with pytest.raises(ValueError):
            _check_required({"primary_issue": blank}, ["primary_issue"])

    # **빈 리스트·딕셔너리는 답이다.** affected_conditions: [] 는 "영향받는 조건 없음",
    # not_judgeable: [] 는 "판정 불가인 조건 없음" 이라는 정당한 답이다.
    # 처음에 이것까지 결측으로 쳤다가 결함 에이전트가 무더기로 실패했다.
    _check_required({"affected_conditions": [], "not_judgeable": []},
                    ["affected_conditions", "not_judgeable"])
    for real in (0, False, "E14", ["E14"], [], {}):
        assert _is_blank(real) is False, f"{real!r} 은 답으로 봐야 한다"


def test_a_key_the_prompt_allows_to_be_null_is_not_missing():
    """**프롬프트가 허용한 답을 코드가 거부하면 계약 불일치다.**

    AGG_SYSTEM 은 출력 형식을 `"process_window": "조건 id 목록 또는 null"` 이라고 적어 둔다.
    결함 없는 조건 구간이 없으면 null 이 정답이다 — S4 는 14조건 전부 언더컷이 있다.
    이걸 결측으로 치면 모델이 재시도를 다 태우고 런 전체가 실패한다
    (2026-09-25 스모크에서 flash-lite 가 8회 연속 null 을 내고 런이 죽었다).

    키가 **없는** 것은 여전히 형식 오류다 — 그건 모델이 형식을 어긴 것이지 답한 게 아니다.
    """
    import pytest
    from harc.llm import _check_required

    # 값이 null 이어도 통과한다
    _check_required({"primary_issue": "undercut", "process_window": None},
                    ["primary_issue", "process_window"], nullable=["process_window"])
    # 키 자체가 없으면 nullable 이어도 형식 오류다
    with pytest.raises(ValueError, match="process_window"):
        _check_required({"primary_issue": "undercut"},
                        ["primary_issue", "process_window"], nullable=["process_window"])
    # nullable 은 지정한 키에만 적용된다
    with pytest.raises(ValueError, match="primary_issue"):
        _check_required({"primary_issue": None, "process_window": None},
                        ["primary_issue", "process_window"], nullable=["process_window"])


def test_the_aggregate_must_name_a_process_window():
    """v3: 공정 창은 반드시 지목한다 — 프롬프트와 호출부가 같이 움직여야 한다.

    v2 는 "조건 id 목록 또는 null" 이었고 multi 의 83% 가 null 을 냈다(초고 3.2 절).
    """
    import inspect
    from harc import pipeline
    src = inspect.getsource(pipeline.run)
    assert 'nullable=["onset_condition"]' in src and 'nullable=["process_window"]' not in src
    assert "또는 null" not in pipeline.AGG_SYSTEM and "반드시" in pipeline.AGG_SYSTEM
    assert '"onset_condition"' in pipeline.AGG_SYSTEM
    assert "반드시" in pipeline.single_system(True) and "반드시" in pipeline.single_system(False)


def test_parallel_cap_is_recorded_and_per_provider(series):
    """google 은 분당 토큰 한도 때문에 동시 3개로 묶는다 (2026-10-02). 결과에는 영향이 없다."""
    from harc.llm import MAX_PARALLEL
    from harc.defects.registry import ALL
    out = pipeline.run(series("S2"), provider="mock", mode="multi", base_dir="data")
    assert out["max_parallel"] == len(ALL)
    out = pipeline.run(series("S2"), provider="mock", mode="multi", base_dir="data", max_parallel=2)
    assert out["max_parallel"] == 2
    assert MAX_PARALLEL["google"] == 3


def test_goal_reaches_the_prompt_and_never_names_a_condition(series, recorded):
    """v4: goal(논문이 내건 목표)은 모델 입력에 들어간다. 조건 id 가 들어 있으면 답을 알려 주는 것이다."""
    import json, pathlib, re
    root = pathlib.Path(__file__).resolve().parent.parent
    for p in sorted((root / "data" / "series").glob("S*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        g = s.get("goal")
        assert g, f"{p.stem} 에 goal 이 없다"
        for c in s["conditions"]:
            assert not re.search(rf"(?<![A-Za-z0-9]){re.escape(c['id'])}(?![0-9])", g), (p.stem, c["id"])
        for w in ("최적", "optim", "가장 좋", "best"):
            assert w not in g, (p.stem, w)
    pipeline.run(series("S4"), provider="mock", mode="multi", base_dir="data")
    from harc.defects.registry import ALL
    agents = {c["agent"]: c["user"] or "" for c in recorded}
    for name in [f"defect:{d.name}" for d in ALL] + ["aggregate", "cause", "next"]:
        assert "AR > 10" in agents[name], f"{name} 프롬프트에 goal 이 없다"
