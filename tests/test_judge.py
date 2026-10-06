"""채점 레이어. 정답은 answers/ 에만 있고 파이프라인 프롬프트로 새지 않아야 한다."""
import json
import pytest
from harc import judge


ALIASES = {"L1": "CHF3 / 20C", "L2": "CHF3 / -20C", "L4": "C3H2F6 / -20C"}


# --- 집합 채점 ---

def test_mixed_correct_and_incorrect_abstains():
    """정답 표현과 오답 표현이 함께 나오면 코드는 기권한다.

    문자열만으로는 'L1 도 공정창이다'(오답)와 'L1 대비 개선'(비교 기준)을 구분할 수 없다.
    outputs_rag/ 채점에서 '오답이 이긴다' 규칙이 만든 오답 4건이 전부 후자였다.
    """
    r = {"correct": ["L4"], "acceptable": [], "incorrect": ["L1"]}
    v, why = judge.grade_set("L4 가 좋지만 L1 도 공정창이다", r)
    assert v is None and "구분 불가" in why


def test_incorrect_alone_is_still_incorrect():
    r = {"correct": ["L4"], "incorrect": ["L1"]}
    v, why = judge.grade_set("공정창은 L1 이다", r)
    assert v == "incorrect" and "L1" in why


def test_excluded_condition_mentioned_as_a_reason_is_not_penalized():
    """'C4 이상에서 식각 중단' 은 C4 를 배제한 것이지 지목한 것이 아니다."""
    r = {"correct": ["C2", "C3"], "incorrect": ["C4", "C5"]}
    v, _ = judge.grade_set("C1 ~ C3 (C4 이상에서 식각 중단 발생)", r)
    assert v is None          # 오답으로 단정하지 않는다


def test_unanticipated_answer_is_ungraded_not_zero():
    """루브릭이 예상 못 한 답은 None 이다. 0 점(오답)과 구분해야 한다."""
    r = {"correct": ["L4"], "incorrect": ["L1"]}
    v, _ = judge.grade_set("판정 불가", r)
    assert v is None
    assert judge.SCORE.get(v) is None


def test_missing_answer_is_ungraded():
    assert judge.grade_set(None, {"correct": ["L4"]})[0] is None


# --- 조건 별칭 ---

def test_condition_matched_by_its_label(): 
    """모델이 'L1' 대신 'CHF3 at 20C' 라고 쓰는 일이 흔하다."""
    assert judge._mentions("CHF3 at 20C (top_bottom_ratio: 4.0)", "L1", ALIASES)


def test_minus_sign_separates_20c_from_minus20c():
    """'20c' 가 '-20c' 안에서 잡히면 L1(20℃)과 L2(-20℃)를 구분 못 한다."""
    assert not judge._mentions("CHF3 를 -20C 에서", "L1", ALIASES)
    assert judge._mentions("CHF3 / -20C 조건", "L2", ALIASES)


def test_condition_id_needs_word_boundary():
    assert not judge._mentions("L10 조건", "L1", ALIASES)
    assert judge._mentions("공정창은 L1 이다", "L1", ALIASES)


def test_label_match_needs_every_token():
    """라벨 토큰이 일부만 나오면 그 조건으로 보지 않는다."""
    assert not judge._mentions("CHF3 를 썼다", "L1", ALIASES)      # 온도 없음


# --- 스키마를 벗어난 런 ---

def test_string_shaped_recommendation_does_not_crash():
    """v1 런에 next.recommendation 이 문자열인 것이 있다 — 버리지 말고 채점한다."""
    run = {"mode": "multi", "aggregate": {"primary_issue": "taper"},
           "next": {"recommendation": "CHF3 로 되돌리세요"}, "cause": {}, "rules": {}}
    got = judge.run_answers(run)
    assert got["q3.direction"] == "CHF3 로 되돌리세요"
    assert got["q1.primary_issue"] == "taper"


def test_single_mode_fields_are_read():
    run = {"mode": "single", "single": {
        "q1": {"primary_issue": "taper", "onset_condition": "L1", "process_window": "L4"},
        "q2": {"hypotheses": [{"mechanism": "측벽 패시베이션"}]},
        "q3": {"variable": "gas", "direction": "C3H2F6 사용"}}}
    got = judge.run_answers(run)
    assert got["q1.process_window"] == "L4"
    assert "측벽" in got["q2.mechanism"]


# --- 요약 ---

def test_mean_excludes_ungraded():
    rows = [{"dir": "d", "verdict": "correct", "score": 1.0},
            {"dir": "d", "verdict": "incorrect", "score": 0.0},
            {"dir": "d", "verdict": None, "score": None}]
    s = judge.summarize(rows, ("dir",))[("d",)]
    assert s["scored"] == 2 and s["ungraded"] == 1
    assert s["mean"] == pytest.approx(0.5)      # 무채점을 0 으로 셌다면 0.33 이 된다
    assert s["wrong"] == 1


# --- 정답 파일 자체 ---

@pytest.mark.parametrize("sid", ["S1", "S2"])
def test_answer_file_is_well_formed(sid):
    a = judge.load_answer(sid)
    assert a and a["series_id"] == sid
    assert a["source"]["paper_id"] and a["authors_conclusion"]
    for q in ("q1", "q2", "q3"):
        assert a[q]
    for key, r in a["q1"].items():
        assert r["correct"], f"{sid}.q1.{key} 에 정답이 없다"
        overlap = set(r["correct"]) & set(r.get("incorrect", []))
        assert not overlap, f"{sid}.q1.{key} 에서 정답과 오답이 겹친다: {overlap}"


def test_s2_bowing_is_incorrect_as_primary_issue():
    """S2 데이터에는 보잉 지표가 없어 네 조건 전부 '판정 불가'다.
    그걸 보고도 주 문제로 꼽는 것은 이 파이프라인이 막으려는 행동이다."""
    pi = judge.load_answer("S2")["q1"]["primary_issue"]
    assert "bowing" in pi["incorrect"] and "bowing" not in pi.get("acceptable", [])


def test_direction_rubric_has_no_bare_chemical_name():
    """맨 화학식을 정답 표현으로 두면 'minimize C3H2F6' 같은 반대 권고도 정답이 된다."""
    for sid in ("S1", "S2"):
        for token in judge.load_answer(sid)["q3"]["direction"]["correct"]:
            assert token.strip() not in ("C3H2F6", "CHF3"), f"{sid}: 맨 화학식 {token}"


def test_answers_never_reach_the_pipeline():
    """정답 디렉터리를 파이프라인 쪽 코드가 읽으면 프롬프트로 샐 수 있다."""
    import pathlib
    src = pathlib.Path(judge.ROOT / "harc")
    for f in src.glob("*.py"):
        if f.name == "judge.py":
            continue
        assert "answers" not in f.read_text(encoding="utf-8"), f"{f.name} 이 answers 를 참조한다"


# --- LLM 판정 경로 ---

class FakeLLM:
    """call_json 을 대신한다. 호출 기록을 남겨 '한 번만 부르는지'를 검사한다."""
    def __init__(self, verdict="correct", why="이유"):
        self.calls, self.verdict, self.why = [], verdict, why

    def __call__(self, agent, provider, system, user, required=(), model=None, **kw):
        self.calls.append({"agent": agent, "system": system, "user": user, "run_id": kw.get("run_id")})
        return {"verdict": self.verdict, "why": self.why}


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    import harc.llm
    f = FakeLLM()
    monkeypatch.setattr(harc.llm, "call_json", f)
    monkeypatch.setattr(judge, "JUDGE_CACHE", tmp_path / "judge_cache.json")
    return f


def _row(**kw):
    base = dict(series="S2", field="q3.direction", answer="C3H2F6 로 가세요",
                verdict=None, score=None, judge="code", why="")
    base.update(kw)
    return base


def test_llm_judges_only_what_code_abstained_on(fake_llm):
    rows = [_row(verdict="correct", score=1.0),          # 코드가 판정함 → 건드리지 않는다
            _row(verdict=None)]                          # 코드가 기권 → LLM
    judge.judge_with_llm(rows, "google")
    assert len(fake_llm.calls) == 1
    assert rows[0]["judge"] == "code" and rows[0]["verdict"] == "correct"
    assert rows[1]["judge"] == "llm:google" and rows[1]["verdict"] == "correct"
    assert rows[1]["score"] == 1.0


def test_keyword_provisional_q2_is_rejudged(fake_llm):
    """Q2 는 코드가 점수를 내도 '잠정' 이므로 LLM 이 다시 본다."""
    rows = [_row(field="q2.mechanism", answer="측벽 패시베이션",
                 verdict="acceptable", score=0.5, judge="keyword(잠정)")]
    judge.judge_with_llm(rows, "google")
    assert len(fake_llm.calls) == 1 and rows[0]["judge"] == "llm:google"


def test_identical_answers_cost_one_call(fake_llm):
    """img off/on 쌍은 답이 같은 경우가 많다 — 같은 답은 한 번만 부른다."""
    rows = [_row(), _row(), _row(answer="C3H2F6  로   가세요")]   # 공백만 다름
    judge.judge_with_llm(rows, "google")
    assert len(fake_llm.calls) == 1
    assert all(r["verdict"] == "correct" for r in rows)


def test_unanswerable_stays_ungraded(fake_llm):
    """'답이 질문에 답하지 않음' 은 오답이 아니라 무채점이다."""
    fake_llm.verdict = "unanswerable"
    rows = [_row()]
    judge.judge_with_llm(rows, "google")
    assert rows[0]["verdict"] is None and rows[0]["score"] is None
    assert "unanswerable" in rows[0]["why"]


def test_unknown_verdict_is_not_scored(fake_llm):
    fake_llm.verdict = "아주 좋음"
    rows = [_row()]
    judge.judge_with_llm(rows, "google")
    assert rows[0]["verdict"] is None and rows[0]["score"] is None


def test_rows_without_an_answer_are_not_sent(fake_llm):
    rows = [_row(answer=None)]
    judge.judge_with_llm(rows, "google")
    assert fake_llm.calls == [] and rows[0]["verdict"] is None


def test_judge_calls_carry_their_own_run_id(fake_llm):
    """실험 런 비용 집계에 판정 호출이 섞이면 안 된다."""
    judge.judge_with_llm([_row()], "google")
    rid = fake_llm.calls[0]["run_id"]
    assert rid and rid.startswith("judge-")


def test_prompt_carries_the_rubric_and_the_answer(fake_llm):
    judge.judge_with_llm([_row()], "google")
    user = fake_llm.calls[0]["user"]
    assert "C3H2F6 로 가세요" in user          # 채점할 답
    assert "저자 결론" in user and "채점 기준" in user
    assert "unanswerable" in fake_llm.calls[0]["system"]


def test_cache_survives_a_second_run(fake_llm):
    judge.judge_with_llm([_row()], "google")
    assert len(fake_llm.calls) == 1
    rows = [_row()]
    assert judge.judge_with_llm(rows, "google") == 0    # 새 호출 0
    assert rows[0]["verdict"] == "correct"


def test_majority_vote_picks_the_winner_and_records_agreement(monkeypatch, tmp_path):
    """심판은 temperature 0 에서도 스스로 뒤집는다 — 다수결로 줄이고 갈린 정도를 남긴다."""
    import harc.llm
    seq = iter(["incorrect", "correct", "correct"])
    def fake(agent, provider, system, user, required=(), model=None, **kw):
        return {"verdict": next(seq), "why": "이유"}
    monkeypatch.setattr(harc.llm, "call_json", fake)
    monkeypatch.setattr(judge, "JUDGE_CACHE", tmp_path / "c.json")
    rows = [_row()]
    n = judge.judge_with_llm(rows, "google", repeat=3)
    assert n == 3
    assert rows[0]["verdict"] == "correct" and rows[0]["agreement"] == "2/3"
    assert "투표" in rows[0]["why"]


def test_repeat_count_is_part_of_the_cache_key(monkeypatch, tmp_path):
    """repeat 를 바꾸면 캐시를 재사용하지 않는다 — 1회 판정과 다수결은 다른 결과다."""
    import harc.llm
    calls = []
    def fake(agent, provider, system, user, required=(), model=None, **kw):
        calls.append(1); return {"verdict": "correct", "why": "이유"}
    monkeypatch.setattr(harc.llm, "call_json", fake)
    monkeypatch.setattr(judge, "JUDGE_CACHE", tmp_path / "c.json")
    judge.judge_with_llm([_row()], "google", repeat=1)
    judge.judge_with_llm([_row()], "google", repeat=3)
    assert len(calls) == 4          # 1 + 3, 재사용 없음


# --- 코드 채점과 LLM 판정을 섞지 않는다 ---

def test_judge_kind_separates_the_three_sources():
    assert judge.judge_kind({"judge": "code"}) == "code"
    assert judge.judge_kind({"judge": "keyword(잠정)"}) == "keyword"
    assert judge.judge_kind({"judge": "llm:google"}) == "llm"


def test_report_never_prints_a_blended_mean(capsys):
    """코드 채점(재현 가능)과 LLM 판정(자기일관성 75%)을 한 평균에 합치면
    신뢰도 차이가 보이지 않는다. 블록을 나누고 합치지 말라고 경고한다."""
    rows = [{"dir": "d", "judge": "code", "verdict": "correct", "score": 1.0},
            {"dir": "d", "judge": "llm:google", "verdict": "incorrect", "score": 0.0}]
    judge._report(rows, ("dir",))
    out = capsys.readouterr().out
    assert "코드 채점" in out and "LLM 판정" in out
    assert "1.00" in out and "0.00" in out     # 각 블록의 평균
    assert "0.50" not in out                   # 섞은 평균은 나오면 안 된다
    assert "합치지 마세요" in out


def test_report_omits_empty_blocks(capsys):
    judge._report([{"dir": "d", "judge": "code", "verdict": "correct", "score": 1.0}], ("dir",))
    out = capsys.readouterr().out
    assert "코드 채점" in out
    assert "LLM 판정" not in out and "합치지 마세요" not in out


def test_multi_onset_is_not_scored_from_the_rules():
    """multi 의 aggregate 스키마에는 onset 이 없다. 모델 답이 없는데 코드가 계산한
    rules[...]["onset"] 을 가져다 채점하면, 조건이 무엇이든 같은 값이 나와 비교에 잡음만 더한다."""
    run = {"mode": "multi", "aggregate": {"primary_issue": "taper"},
           "rules": {"etch_stop": {"onset": "C4"}}, "cause": {}, "next": {}}
    assert judge.run_answers(run)["q1.onset_condition"] is None


def test_single_onset_still_comes_from_the_model():
    run = {"mode": "single", "single": {"q1": {"onset_condition": "L1"}, "q2": {}, "q3": {}}}
    assert judge.run_answers(run)["q1.onset_condition"] == "L1"


# --- 조건 집합 채점 (공정창·시작 조건) ---
# 공정창은 조건의 집합인데 모델은 구간으로 답한다. 문자열 매칭만으로는 대부분 무채점이 됐다.

C5 = {"C1": 0.40, "C2": 0.47, "C3": 0.56, "C4": 0.65, "C5": 0.75}
L4 = {"L1": "CHF3 / 20C", "L2": "CHF3 / -20C", "L3": "C3H2F6 / 20C", "L4": "C3H2F6 / -20C"}


@pytest.mark.parametrize("text,conds,want", [
    ("C2", C5, {"C2"}),
    ("C1 ~ C3", C5, {"C1", "C2", "C3"}),
    ("C1 to C3 (with C2 being the optimal balance)", C5, {"C1", "C2", "C3"}),
    ("L1 (CHF3 / 20C) ~ L4 (C3H2F6 / -20C)", L4, {"L1", "L2", "L3", "L4"}),
    ("C3H2F6 gas and lower chiller temperature (-20C) improves", L4, {"L4"}),
    ("CHF3 at 20C (top_bottom_ratio: 4.0)", L4, {"L1"}),
])
def test_parse_conditions(text, conds, want):
    assert judge.parse_conditions(text, conds) == want


def test_excluded_clause_is_not_counted():
    """'C4 이상에서 식각 중단' 의 C4 는 공정창이 아니라 공정창이 끝나는 이유다."""
    assert judge.parse_conditions("C1 ~ C3 (C4 이상에서 식각 중단 발생)", C5) == {"C1", "C2", "C3"}


def test_comparison_clause_is_not_counted():
    """'L1 대비' 의 L1 은 비교 기준이지 지목이 아니다. 본문의 L4 는 살아야 한다."""
    assert judge.parse_conditions("L4 (CHF3 / 20C 대비 C3H2F6 및 -20C 적용 시 개선)", L4) == {"L4"}


def test_declining_to_answer_stays_ungraded():
    r = {"correct": ["L4"], "acceptable": ["L3", "L4"], "incorrect": ["L1"]}
    v, why = judge.grade_conditions("판정 불가 (Bowing 데이터 부재)", r, L4)
    assert v is None and "뽑을 수 없음" in why


@pytest.mark.parametrize("text,want", [
    ("L4", "correct"),                       # 정확히 정답 집합
    ("L3 ~ L4", "acceptable"),               # 정답을 포함하되 더 넓음
    ("L1 (CHF3 / 20C) ~ L4 (C3H2F6 / -20C)", "incorrect"),   # 오답 조건 포함
    ("L3", "acceptable"),                    # 허용 집합 안
])
def test_grade_conditions(text, want):
    r = {"correct": ["L4"], "acceptable": ["L3", "L4"], "incorrect": ["L1"]}
    assert judge.grade_conditions(text, r, L4)[0] == want


def test_rubric_marks_condition_fields():
    """공정창·시작 조건은 집합 채점을 쓰도록 표시돼 있어야 한다."""
    for sid in ("S1", "S2"):
        q1 = judge.load_answer(sid)["q1"]
        for k in ("process_window", "onset_condition"):
            assert q1[k].get("judge") == "conditions", (sid, k)


def test_real_range_answers_are_now_graded():
    """저장소 실제 런: 구간으로 답한 것이 무채점으로 버려지지 않아야 한다."""
    rows = [r for r in judge.judge_dir(["outputs_rag_t07"])
            if r["field"] == "q1.process_window" and r["answer"]
            and "~" in str(r["answer"])]
    assert rows, "구간 답이 있어야 이 테스트가 의미 있다"
    assert all(r["score"] is not None for r in rows), \
        [r["answer"] for r in rows if r["score"] is None]
