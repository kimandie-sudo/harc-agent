"""조건 id 매칭 — 공정창·시작 조건 채점이 기대는 부분. API 키 없이 돈다."""


# ── 조건 id 경계 (2026-09-25) ────────────────────────────────────────────
def _cond(series):
    import json, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    return json.loads((root / "answers" / f"{series}.json").read_text(encoding="utf-8"))["conditions"]


def test_letter_only_condition_id_is_not_a_substring():
    """S3 의 기준 조건 id 는 `R` 이다. "RF" 안의 R 을 지목으로 세면 안 된다.

    예전 판은 숫자 없는 id 를 부분 문자열로 찾아서 "…RF 위상차가 180도(P180)인 조건"
    이라는 **정답**을 R 포함 → 오답으로 0점 처리했다. 장황한 답일수록 걸려서
    RAG on 쪽으로 치우친 편향이었다.
    """
    from harc.judge import _mentions, grade_conditions
    import json, pathlib
    c = _cond("S3")
    for text in ("RF 위상차", "etch rate", "bottom ratio", "top/bottom ratio"):
        assert not _mentions(text, "R", c), text
    for text in ("R ~ P90", "(R)", "R 조건", "R, P90"):
        assert _mentions(text, "R", c), text
    root = pathlib.Path(__file__).resolve().parent.parent
    rub = json.loads((root / "answers" / "S3.json").read_text(encoding="utf-8"))["q1"]["onset_condition"]
    v, _ = grade_conditions("상·하부 전극 간 RF 위상차(Δphase)가 180도(P180)인 조건", rub, c)
    assert v == "correct"


def test_condition_id_after_a_word_is_found():
    """`_norm` 이 공백을 지우면 "neck P90" 이 "neckp90" 이 되어 경계가 사라진다.
    조건 id 는 공백을 살린 본문에서 찾아야 한다."""
    from harc.judge import _mentions
    assert _mentions("neck P90", "P90", _cond("S3"))
    assert _mentions("condition C2", "C2", _cond("S1"))


def test_digit_ids_still_do_not_bleed_into_longer_ids():
    from harc.judge import _mentions
    c = _cond("S4")
    assert not _mentions("E10", "E1", c)
    assert _mentions("E1, E14", "E1", c)
    assert not _mentions("-20C", "L2", _cond("S2"))


def test_numeric_label_tokens_need_a_right_boundary():
    """'0.4'(S1 C1)가 '0.47' 안에서, '3'(S5 D3)이 '30' 안에서 잡히면 안 된다.

    예전 판은 숫자 토큰의 앞 경계만 봐서, S1 의 정답 그대로인 "C2 (0.47)" 을 {C1, C2} 로 읽고
    정답이 아니라 절반으로 채점했다 (2026-09-29).
    """
    from harc.judge import parse_conditions, grade_conditions
    import json, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    a1 = json.loads((root / "answers" / "S1.json").read_text(encoding="utf-8"))
    assert parse_conditions("C2 (0.47)", a1["conditions"]) == {"C2"}
    assert grade_conditions("0.47", a1["q1"]["process_window"], a1["conditions"])[0] == "correct"
    assert parse_conditions("0.4", a1["conditions"]) == {"C1"}
    a5 = json.loads((root / "answers" / "S5.json").read_text(encoding="utf-8"))
    assert parse_conditions("D7 (증착 7 s / 식각 30 s)", a5["conditions"]) == {"D7"}
    # 앞 경계는 그대로 — 20C 가 -20C 안에서 잡히면 안 된다
    a2 = json.loads((root / "answers" / "S2.json").read_text(encoding="utf-8"))
    assert parse_conditions("L4 (C3H2F6 / -20C)", a2["conditions"]) == {"L4"}


def _ans(sid):
    import json, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    return json.loads((root / "answers" / f"{sid}.json").read_text(encoding="utf-8"))


def test_primary_issue_in_korean_is_graded():
    """모델은 주 결함을 한국어로 답하는 일이 많다 — 예전에는 전부 무채점이었다 (2026-10-01)."""
    from harc.judge import grade_set
    r6 = _ans("S6")["q1"]["primary_issue"]
    assert grade_set("식각 중단 및 CD 편차", r6)[0] == "correct"
    assert grade_set("Low O2 flow causes etch stop", r6)[0] == "correct"
    assert grade_set("cd_deviation", r6)[0] == "correct"           # 모듈 이름은 그대로
    r1 = _ans("S1")["q1"]["primary_issue"]
    assert grade_set("C1의 보잉, C3의 테이퍼, C4~C5의 식각 중단", r1)[0] == "correct"
    # 오답 결함이 함께 나오면 여전히 기권한다
    r8 = _ans("S8")["q1"]["primary_issue"]
    assert grade_set("Bowing, Mask Erosion, and CD Deviation", r8)[0] is None


def test_label_symbols_ranges_and_alternative_labels():
    """S6·S7 에서 드러난 라벨 매칭 한계 셋 (2026-10-01)."""
    from harc.judge import parse_conditions
    c6 = _ans("S6")["conditions"]
    assert parse_conditions("ICP O2 flow 20–22.5 sccm", c6) == {"I20", "I22"}   # '·' 없음 + 범위 대시
    assert parse_conditions("ICP mode: 20–22.5 sccm", c6) == {"I20", "I22"}     # 짧은 라벨
    assert parse_conditions("CCP 200 sccm", c6) == {"C200"}
    c7 = _ans("S7")["conditions"]
    assert parse_conditions("Bias 100V ~ 200V", c7) == {"B100", "B200"}        # 영어 라벨
    assert parse_conditions("바이어스 300 V", c7) == {"B300"}
    # 범위 대시를 허용해도 음수 부호는 그대로 막는다
    c2 = _ans("S2")["conditions"]
    assert parse_conditions("L4 (C3H2F6 / -20C)", c2) == {"L4"}
    assert parse_conditions("20-22.5", {"A": "22.5"}) == {"A"}
    assert parse_conditions("(-22.5)", {"A": "22.5"}) == set()


def test_onset_is_graded_only_when_the_primary_issue_is_the_authors():
    """S8: 마스크 침식을 주 결함으로 고른 모델은 그 시작 조건(−48 °C)을 답한다.
    저자 결함(네킹) 기준으로 깎으면 같은 실수를 두 번 깎는다 (2026-10-01)."""
    from harc.judge import grade_run
    a = _ans("S8")
    def run(issue, onset):
        return {"mode": "single", "single": {"q1": {"primary_issue": issue, "onset_condition": onset,
                                                    "process_window": "T4"}}}
    v = {r["field"]: r["verdict"] for r in grade_run(run("마스크 침식", "T3"), a)}
    assert v["q1.onset_condition"] is None
    v = {r["field"]: r["verdict"] for r in grade_run(run("네킹", "T3"), a)}
    assert v["q1.onset_condition"] == "incorrect"
    v = {r["field"]: r["verdict"] for r in grade_run(run("necking", "T1 ~ T2"), a)}
    assert v["q1.onset_condition"] == "correct"


def test_multi_v3_onset_comes_from_the_aggregate():
    from harc.judge import run_answers
    v2 = {"mode": "multi", "aggregate": {"primary_issue": "x", "process_window": None}}
    v3 = {"mode": "multi", "aggregate": {"primary_issue": "x", "onset_condition": "T1", "process_window": "T4"}}
    assert run_answers(v2)["q1.onset_condition"] is None
    assert run_answers(v3)["q1.onset_condition"] == "T1"
