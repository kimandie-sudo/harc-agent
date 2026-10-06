"""채점 레이어. 런 JSON 을 answers/ 의 정답 루브릭과 대조한다.

  python -m harc.judge outputs_rag                      # 코드 채점 (API 키 불필요)
  python -m harc.judge outputs_v2 outputs_rag --csv 점수.csv
  python -m harc.judge outputs --by dir model mode

--llm 을 주면 **코드가 기권한 항목(무채점)과 Q2** 만 LLM 이 판정한다.
코드가 판정한 항목은 건드리지 않는다 — 재현 가능한 쪽을 우선한다.
판정 주체는 결과의 judge 열에 남는다 (code / keyword(잠정) / llm:<provider>).
LLM 이 "unanswerable" 이라고 하면 무채점으로 남긴다 (오답과 구분).

정답은 answers/<series>.json 에만 있고 파이프라인은 이 모듈을 import 하지 않는다.
모델 프롬프트에 정답이 섞일 경로가 없어야 하므로, 여기서도 런을 읽기만 한다.

채점 기준 (answers/ 의 correct / acceptable / incorrect 집합):
  correct → 1.0,  acceptable → 0.5,  incorrect → 0.0,  어느 쪽도 아니면 None(무채점)
None 은 0 점이 아니다. 루브릭이 그 답을 예상하지 못한 것이므로 평균에서 빼고 따로 센다.
"""
import argparse, csv, json, pathlib, re, sys, collections

ROOT = pathlib.Path(__file__).resolve().parent.parent
ANSWER_DIR = ROOT / "answers"

SCORE = {"correct": 1.0, "acceptable": 0.5, "incorrect": 0.0}


def load_answer(series_id, answer_dir=None):
    p = pathlib.Path(answer_dir or ANSWER_DIR) / f"{series_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# 같은 뜻인데 글자가 다른 기호. 온도 스윕(S8)에서 모델이 "−63 ℃" 로, 라벨이 "-63 °C" 로 쓰면
# 같은 조건을 못 찾는다 (2026-09-30). 유니코드 마이너스·대시는 '-' 로, ℃ 는 °C 로 맞춘다.
_SAME = str.maketrans({"−": "-", "–": "-", "‒": "-", "℃": "°C", "˚": "°"})


def _unify(s):
    return str(s).translate(_SAME)


def _norm(s):
    """대소문자·공백·따옴표를 지운 비교용 문자열."""
    return re.sub(r"[\s\"'`]+", "", _unify(s)).lower()


def _tok_in(text_norm, token):
    """본문 안에서 토큰 하나를 찾는다. 숫자 토큰은 공백을 살린 본문을 받는다(_mentions 참고).

    '20c' 가 '-20c' 안에서 잡히면 L1(20℃)과 L2(-20℃)를 구분하지 못하므로
    숫자로 시작하는 토큰은 앞에 부호·숫자·소수점이 오지 않을 때만 인정한다.

    **뒤쪽 경계도 본다.** 예전에는 앞만 봐서 '0.4'(S1 의 C1)가 '0.47' 안에서, '3'(S5 의 증착 3 s)이
    '30' 안에서 잡혔다. S1 에서 정답 그대로인 "C2 (0.47)" 이 {C1, C2} 로 읽혀 정답이 아니라 절반으로
    채점됐다 (2026-09-29 발견). 숫자 토큰 뒤에 숫자나 '.숫자' 가 오면 다른 수다.
    """
    t = _norm(token)
    if not t:
        return False
    if re.match(r"^\d", t):
        # 앞의 '-' 가 숫자 뒤에 붙어 있으면 부호가 아니라 범위 대시다 ("20–22.5 sccm" 의 22.5, 2026-10-01)
        return re.search(rf"(?<![\d.])(?<!(?<![\d])-){re.escape(t)}(?!\d|\.\d)", text_norm) is not None
    if re.match(r"^-\d", t):                   # 음수: '-63' 이 '-630' 이나 '1-63' 안에서 잡히면 안 된다
        return re.search(rf"(?<![\d.]){re.escape(t)}(?!\d|\.\d)", text_norm) is not None
    return t in text_norm


# 주 결함은 루브릭에 모듈 이름(영문)으로 적혀 있지만, 모델은 한국어로 답하는 일이 많다
# ("식각 중단 및 CD 편차"). 모듈 이름이 needle 일 때만 이 표를 쓴다 (2026-10-01).
DEFECT_ALIASES = {
    "bowing": ["보잉", "bowing", "bow index"],
    "taper": ["테이퍼", "taper"],
    "necking": ["네킹", "넥킹", "necking"],
    "undercut": ["언더컷", "undercut"],
    "sidewall_angle": ["측벽각", "측벽 각도", "sidewall angle"],
    "etch_stop": ["식각 중단", "식각 정지", "etch stop", "etch-stop"],
    "mask_erosion": ["마스크 침식", "mask erosion"],
    "cd_deviation": ["CD 편차", "cd deviation"],
    "depth_loss": ["식각 깊이 감소", "깊이 감소", "depth loss"],
    "low_aspect_ratio": ["종횡비 부족", "low aspect ratio"],
}


def _labels(label):
    """조건 라벨은 문자열 하나 또는 여러 표현의 목록이다 (예: ['바이어스 100 V', 'bias 100 V'])."""
    if not label:
        return []
    return [str(x) for x in label] if isinstance(label, (list, tuple)) else [str(label)]


def _mentions(text, needle, aliases=None):
    r"""자유 서술 안에 needle 이 들어 있는가.

    조건 id(C1, L4, R)는 단어 경계로 본다. 그리고 aliases 에 그 조건의 라벨이 있으면
    라벨의 토큰이 전부 나와도 인정한다 — 모델이 'L1' 대신 'CHF3 at 20C' 라고 쓰는 일이 흔하다.

    **숫자 없는 조건 id 도 조건 id 다.** 예전에는 `[a-z]\d+` 꼴만 id 로 보고 나머지는
    부분 문자열로 찾았다. S3 의 기준 조건 id 가 `R` 이라, 답에 영문 r 이 하나라도 있으면
    ("RF 위상차", "ratio", "etch rate") R 을 지목한 것으로 쳤고 R 은 오답 집합이라 0점이 됐다.
    "…RF 위상차가 180도(P180)인 조건" 이라는 **정답**이 0점을 받았다 (2026-09-25 발견).
    장황한 답일수록 걸리므로 무작위 잡음이 아니라 RAG on 쪽으로 치우친 편향이었다.
    aliases(조건 표)에 키로 있는 needle 은 모양과 무관하게 조건 id 로 본다.

    **조건 id 는 공백을 살린 본문에서 찾는다.** `_norm` 은 공백을 지우므로 단어 경계가
    사라진다 — "neck P90" 이 "neckp90" 이 되어 앞 글자 k 때문에 P90 을 못 찾았고,
    "condition C2" 도 같은 이유로 놓쳤다. 경계를 보려면 경계가 남아 있어야 한다.
    """
    t, n = _norm(text), _norm(needle)
    if not n:
        return False
    is_id = re.fullmatch(r"[a-z]\d+", n) or str(needle).strip() in (aliases or {})
    if is_id:                                  # C1, L4, R 같은 조건 id
        tw = re.sub(r"[\"'`]+", "", _unify(text)).lower()  # 공백은 살린다
        # 숫자로 끝나면 뒤에 숫자만 막는다(E1 ≠ E10). 글자로 끝나면 글자도 막는다(R ≠ RF).
        tail = r"(?![0-9])" if n[-1].isdigit() else r"(?![a-z0-9])"
        if re.search(rf"(?<![a-z0-9]){re.escape(n)}{tail}", tw):
            return True
        for label in _labels((aliases or {}).get(str(needle).strip())):
            # 글자·숫자가 없는 토큰('·' 등)은 버린다 — 모델은 'ICP · O2' 를 'ICP O2' 로 쓴다 (S6, 2026-10-01)
            toks = [x for x in re.split(r"[\s/,()]+", label) if re.search(r"\w", x)]
            # 숫자 토큰은 공백을 살린 본문에서 찾는다 — "O2 20 sccm" 이 "o220sccm" 으로 붙으면
            # 20 앞에 숫자 2 가 와서 앞 경계에 걸린다 (S6, 2026-09-29)
            if toks and all(_tok_in(tw if re.match(r"^-?\d", _norm(x)) else t, x) for x in toks):
                return True
        return False
    if str(needle).strip() in DEFECT_ALIASES:
        return n in t or any(_norm(a) in t for a in DEFECT_ALIASES[str(needle).strip()])
    return n in t


# ---------------- 조건 집합 채점 (공정창·시작 조건) ----------------
# 공정창은 '조건의 집합' 이다. 그런데 모델은 "C1 ~ C3", "L1 (…) ~ L4 (…)", "0.47 ~ 0.56" 처럼
# 구간으로 답한다. 집합으로 파싱하지 않고 문자열만 맞춰 보면 대부분 무채점이 됐다
# (세 배치 합쳐 13건 중 11건이 구간 표현이었다).
#
# 배제 절도 가려야 한다. "C1 ~ C3 (C4 이상에서 식각 중단 발생)" 에서 C4 는 공정창이 아니라
# 공정창이 끝나는 이유다. 이것을 지목으로 세면 오답으로 잘못 단정한다.

_EXCLUDE_MARKERS = ("이상", "이후", "부터", "넘으", "초과", "대비", "제외", "보다",
                    "compared", "except", "beyond", "above", "onward")
_RANGE = r"(?:~|∼|-|–|—|\bto\b|내지|에서)"


def parse_conditions(text, conditions):
    """자유 서술에서 조건 id 집합을 뽑는다. 구간은 펼치고, 배제 절은 뺀다.

    conditions 는 answers/<시리즈>.json 의 {id: 라벨/값} — **순서가 조건 순서**다.
    """
    if text is None or not conditions:
        return set()
    t = str(text)
    order = list(conditions)

    # 배제 절(괄호 안에 '이상'·'대비' 같은 표지가 있는 것)은 본문에서 들어낸다.
    # 들어낸 뒤에 찾으므로 따로 차감하지 않는다 — 차감까지 하면 본문에 제대로 적힌 조건까지 지워진다.
    body = re.sub(r"\(([^)]*)\)", lambda m: "" if any(x in m.group(1) for x in _EXCLUDE_MARKERS)
                  else m.group(0), t)

    got = {cid for cid in order if _mentions(body, cid, conditions)}

    # 구간: 본문에 "<A> … ~ … <B>" 형태가 있으면 사이를 채운다
    ids = "|".join(re.escape(c) for c in order)
    for a, b in re.findall(rf"({ids})\b[^{ids}]{{0,40}}?{_RANGE}[^{ids}]{{0,40}}?({ids})\b", body):
        i, j = order.index(a), order.index(b)
        got |= set(order[min(i, j):max(i, j) + 1])

    return got


def grade_conditions(answer_text, rubric, conditions):
    """조건 집합으로 채점한다.

    오답 조건이 하나라도 들어가면 오답. 정확히 정답 집합이면 정답.
    정답을 포함하되 더 넓거나, 허용 집합 안에 들어가면 절반.
    하나도 못 뽑으면 무채점 ('판정 불가' 선언이나 질문에 답하지 않은 경우).
    """
    if answer_text is None:
        return None, "답 없음"
    got = parse_conditions(answer_text, conditions)
    if not got:
        return None, "조건을 뽑을 수 없음 (판정 불가 선언이거나 질문에 답하지 않음)"
    correct = set(rubric.get("correct", []))
    ok = set(rubric.get("acceptable", [])) or correct
    bad = got & set(rubric.get("incorrect", []))
    if bad:
        return "incorrect", f"공정창에 오답 조건 {sorted(bad)} 포함 (뽑힌 집합 {sorted(got)})"
    if got == correct:
        return "correct", f"정답 집합 {sorted(got)}"
    if correct and correct <= got:
        return "acceptable", f"정답 {sorted(correct)} 을 포함하되 더 넓음 {sorted(got)}"
    if got <= ok:
        return "acceptable", f"허용 집합 안 {sorted(got)}"
    return None, f"루브릭 밖의 조건 집합 {sorted(got)}"


def grade_set(answer_text, rubric, aliases=None):
    """자유 서술 하나를 correct/acceptable/incorrect 집합과 대조한다.

    정답 표현과 오답 표현이 **함께** 나오면 코드는 판정하지 않고 기권한다(None).
    문자열만으로는 두 경우를 구분할 수 없기 때문이다:

      "L4 가 좋지만 L1 도 공정창이다"          → 진짜 오답
      "L4 (CHF3 / 20C 대비 개선)"              → 정답. L1 을 비교 기준으로 들었을 뿐
      "C1 ~ C3 (C4 이상에서 식각 중단)"         → 정답. C4 를 배제 이유로 들었을 뿐

    처음에는 '오답이 이기는' 규칙이었는데, outputs_rag/ 를 채점해 보니 그 규칙이 만든
    오답 4건이 **전부** 뒤의 두 경우였다. 틀리게 단정하느니 기권하고 --llm 에 넘긴다.
    """
    if answer_text is None:
        return None, "답 없음"
    hit = lambda key: [x for x in rubric.get(key, []) if _mentions(answer_text, x, aliases)]
    bad, good, ok = hit("incorrect"), hit("correct"), hit("acceptable")
    if bad and (good or ok):
        return None, (f"정답 표현 {good or ok} 과 오답 표현 {bad} 이 함께 있음 — "
                      "배제·대조로 언급한 것인지 코드로는 구분 불가")
    if bad:
        return "incorrect", f"오답 표현 {bad}"
    if good:
        return "correct", f"정답 표현 {good}"
    if ok:
        return "acceptable", f"허용 표현 {ok}"
    return None, "루브릭의 어느 집합에도 없음"


def _pick(obj, *keys):
    """obj[key] 를 꺼내되, obj 가 dict 가 아니면 obj 자체를 답으로 본다.

    모델이 스키마를 지키지 않고 {"recommendation": "비를 낮추세요"} 처럼 문자열 하나로
    답하는 일이 흔하다 (v1 런에 특히 많다). 그것도 채점 대상이므로 버리지 않는다.
    """
    if obj in (None, "", [], {}):
        return None
    if not isinstance(obj, dict):
        return _text(obj)
    return _join(*(obj.get(k) for k in keys))


def _text(v):
    if v in (None, "", [], {}):
        return None
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _sub(obj, key):
    """중첩 dict 한 단계. obj 가 dict 가 아니면 그대로 돌려준다 (스키마 이탈 허용)."""
    return obj.get(key) if isinstance(obj, dict) else obj


def _hyps(obj):
    """hypotheses 목록을 꺼낸다. dict 가 아니거나 키가 없으면 obj 자체를 본다."""
    if isinstance(obj, dict):
        h = obj.get("hypotheses")
        return h if isinstance(h, list) else ([obj] if h is None else h)
    return obj if isinstance(obj, list) else None


def run_answers(run):
    """런 JSON 에서 채점 대상 문자열을 꺼낸다. multi 와 single 의 위치가 다르다.

    모델이 스키마를 벗어난 모양으로 답해도 죽지 않고 원문을 그대로 채점에 넘긴다.
    """
    if run.get("mode") == "multi":
        agg = run.get("aggregate")
        rec = _sub(run.get("next"), "recommendation")
        return {
            "q1.primary_issue": _pick(agg, "primary_issue"),
            "q1.onset_condition": _first_onset(run),
            "q1.process_window": _pick(agg, "process_window"),
            "q2.mechanism": _mech_text(_hyps(run.get("cause"))),
            "q3.variable": _pick(rec, "variable"),
            "q3.direction": _pick(rec, "direction", "suggested_levels"),
        }
    s = run.get("single") or {}
    q1, q3 = _sub(s, "q1"), _sub(s, "q3")
    return {
        "q1.primary_issue": _pick(q1, "primary_issue"),
        "q1.onset_condition": _pick(q1, "onset_condition"),
        "q1.process_window": _pick(q1, "process_window"),
        "q2.mechanism": _mech_text(_hyps(_sub(s, "q2"))),
        "q3.variable": _pick(q3, "variable"),
        "q3.direction": _pick(q3, "direction", "suggested_levels"),
    }


def _first_onset(run):
    """multi 의 onset. **v2 까지는 모델 답이 없다** → 채점하지 않는다.

    v2 의 aggregate 스키마는 ranking/primary_issue/process_window/tradeoffs 뿐이라 onset 을 묻지 않는다.
    예전에는 rules[...]["onset"](코드가 계산한 값)을 가져다 채점했는데, 그것은 모델의 답이 아니라
    코드의 출력이다. 조건이 무엇이든 같은 값이 나오므로 rag on/off 비교에 잡음만 더한다.
    v3 부터는 aggregate 가 onset_condition 을 직접 답한다.
    """
    agg = run.get("aggregate")
    return _pick(agg, "onset_condition") if isinstance(agg, dict) else None


def _join(*parts):
    xs = [json.dumps(p, ensure_ascii=False) if isinstance(p, (list, dict)) else str(p)
          for p in parts if p not in (None, "", [], {})]
    return " / ".join(xs) if xs else None


def _mech_text(hyps):
    if not hyps:
        return None
    if isinstance(hyps, str):
        return hyps
    out = []
    for h in hyps:
        if isinstance(h, dict):
            out.append(" ".join(str(h.get(k, "")) for k in ("issue", "mechanism", "predicted_trend")))
        elif h:
            out.append(str(h))
    return " | ".join(x for x in out if x.strip()) or None


def grade_run(run, answer):
    """런 하나를 채점한다. Q2 는 키워드 기준의 잠정 점수만 내고 judge='llm' 으로 표시한다."""
    got = run_answers(run)
    rows = []
    for q in ("q1", "q2", "q3"):
        for key, rubric in (answer.get(q) or {}).items():
            if not isinstance(rubric, dict):
                continue
            field = f"{q}.{key}"
            text = got.get(field)
            if rubric.get("judge") == "llm":
                verdict, why = _grade_mechanism(text, rubric)
                rows.append(dict(field=field, verdict=verdict, score=SCORE.get(verdict),
                                 why=why, judge="keyword(잠정)", answer=text))
            elif rubric.get("judge") == "conditions":
                verdict, why = grade_conditions(text, rubric, answer.get("conditions") or {})
                # 시작 조건은 '그 문제' 가 시작되는 곳이다. 모델이 주 결함을 저자와 다르게 골랐으면
                # 자기 결함의 시작 조건을 답하므로, 저자 결함 기준의 루브릭으로 채점하면 같은 실수를
                # 두 번 깎는다 (S8, 2026-10-01). 주 결함이 정답일 때만 채점한다.
                if key == "onset_condition" and verdict is not None:
                    pi = next((r for r in rows if r["field"] == "q1.primary_issue"), None)
                    if pi is not None and pi["verdict"] != "correct":
                        verdict, why = None, (f"주 결함이 저자와 달라({pi['verdict'] or '무채점'}) "
                                              f"시작 조건은 채점하지 않음 — 원래 판정 {verdict}: {why}")
                rows.append(dict(field=field, verdict=verdict, score=SCORE.get(verdict),
                                 why=why, judge="code", answer=text))
                continue
            else:
                verdict, why = grade_set(text, rubric, answer.get("conditions"))
                rows.append(dict(field=field, verdict=verdict, score=SCORE.get(verdict),
                                 why=why, judge="code", answer=text))
    return rows


def _grade_mechanism(text, rubric):
    """Q2 의 코드 기준 잠정 채점. 최종 판정은 LLM 이 한다 (--q2)."""
    if text is None:
        return None, "답 없음"
    bad = [x for x in rubric.get("incorrect_claims", []) if _mentions(text, x)]
    if bad:
        return "incorrect", f"오답 주장 {bad}"
    must = [x for x in rubric.get("must_contain_any", []) if _mentions(text, x)]
    should = [x for x in rubric.get("should_contain", []) if _mentions(text, x)]
    if must and should:
        return "correct", f"핵심 {must} + 세부 {should}"
    if must:
        return "acceptable", f"핵심 {must} 만 (세부 근거 없음)"
    return "incorrect", "핵심 키워드 없음"


# ---------------- LLM 판정 ----------------
# 코드는 재현 가능하지만 표현을 다 덮지 못한다. 실제로 q3.direction 은 절반이 무채점이었다.
# 그래서 코드가 기권한 항목과 judge="llm" 항목만 LLM 에게 넘긴다. 코드가 판정한 것은 건드리지 않는다.

JUDGE_CACHE = ROOT / "logs" / "judge_cache.json"

FIELD_QUESTION = {
    "q1.primary_issue": "이 공정 시리즈에서 가장 중대한 문제는 무엇인가",
    "q1.onset_condition": "그 문제가 시작되는 조건은 어디인가",
    "q1.process_window": "문제가 없는 조건 구간(공정 창)은 어디인가",
    "q2.mechanism": "관측된 변화를 만든 공정 기제는 무엇인가",
    "q3.variable": "다음 실험에서 바꿀 변수는 무엇인가",
    "q3.direction": "그 변수를 어느 방향으로 바꿔야 하는가",
}

JUDGE_SYSTEM = """당신은 반도체 식각(고종횡비 콘택) 공정 분석 답안을 채점합니다.
아래 '채점 기준'만 근거로 삼으세요. 기준에 없는 잣대를 새로 만들지 마세요.
반드시 JSON 객체 하나만 출력하세요.

판정값:
- "correct"      기준의 정답에 해당한다
- "acceptable"   틀렸다고 할 수 없으나 정답보다 약하다
- "incorrect"    기준의 오답에 해당하거나 저자 결론과 배치된다
- "unanswerable" 답이 그 질문에 답하지 않는다 (값만 다시 나열, '판정 불가' 선언 등).
                 틀린 것이 아니라 채점 대상이 아니라는 뜻이다. 답이 애매하면 이것을 고르세요.

출력 키: {"verdict": , "why": "한 문장"}"""


def _judge_prompt(field, answer_text, rubric, ans):
    parts = [f"질문: {FIELD_QUESTION.get(field, field)}",
             f"시리즈: {ans['series_id']}",
             f"저자 결론: {ans['authors_conclusion']}", "", "채점 기준:"]
    for key, label in (("correct", "정답"), ("acceptable", "허용"), ("incorrect", "오답")):
        if rubric.get(key):
            parts.append(f"  {label}: {rubric[key]}")
    if rubric.get("reference"):
        parts.append(f"  정답 서술: {rubric['reference']}")
    if rubric.get("incorrect_claims"):
        parts.append(f"  오답 주장: {rubric['incorrect_claims']}")
    if rubric.get("note"):
        parts.append(f"  참고: {rubric['note']}")
    if ans.get("conditions"):
        parts.append(f"  조건 이름: {ans['conditions']}")
    parts += ["", "채점할 답:", '"""', str(answer_text), '"""']
    return "\n".join(parts)


def _cache_load():
    if JUDGE_CACHE.exists():
        try:
            return json.loads(JUDGE_CACHE.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def _cache_save(cache):
    JUDGE_CACHE.parent.mkdir(exist_ok=True)
    JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def judge_with_llm(rows, provider, model=None, answer_dir=None, redo_all=False, repeat=1):
    """코드가 기권한 항목(verdict=None)과 judge='llm' 항목을 LLM 으로 판정한다.

    같은 (시리즈·항목·답) 은 한 번만 부른다 — img off/on 쌍은 답이 같은 경우가 많다.
    결과는 logs/judge_cache.json 에 남아 다시 돌릴 때 호출이 들지 않는다.

    repeat>1 이면 같은 답을 여러 번 판정해 **다수결**을 쓴다. 심판은 temperature 0 에서도
    스스로 뒤집는다 (이 저장소 실측: Q2 표본 12개 중 3개가 두 번 판정에서 갈렸다 = 75% 일관).
    각 행의 agreement 열에 '다수/전체' 가 남으므로, 갈린 항목을 걸러서 읽을 수 있다.
    """
    from .llm import call_json
    import datetime as _dt

    cache = _cache_load()
    run_id = "judge-" + _dt.datetime.now().strftime("%m%d-%H%M%S")
    answers = {}
    calls = 0
    for r in rows:
        needs = redo_all or r["verdict"] is None or r["judge"].startswith("keyword")
        if not needs or r["answer"] is None:
            continue
        ans = answers.setdefault(r["series"], load_answer(r["series"], answer_dir))
        if ans is None:
            continue
        q, key = r["field"].split(".", 1)
        rubric = (ans.get(q) or {}).get(key)
        if not isinstance(rubric, dict):
            continue
        ck = json.dumps([provider, model, r["series"], r["field"], _norm(r["answer"]), repeat],
                        ensure_ascii=False)
        if ck not in cache:
            votes, whys = [], []
            for _ in range(max(1, repeat)):
                out = call_json(f"judge:{r['field']}", provider=provider, model=model,
                                system=JUDGE_SYSTEM,
                                user=_judge_prompt(r["field"], r["answer"], rubric, ans),
                                required=["verdict", "why"], run_id=run_id)
                votes.append(str(out.get("verdict", "")).strip().lower())
                whys.append(str(out.get("why", ""))[:200])
                calls += 1
            top, n_top = collections.Counter(votes).most_common(1)[0]
            cache[ck] = {"verdict": top if top in SCORE else None,
                         "why": whys[votes.index(top)],
                         "raw_verdict": top, "votes": votes,
                         "agreement": f"{n_top}/{len(votes)}"}
            _cache_save(cache)
        c = cache[ck]
        r.update(verdict=c["verdict"], score=SCORE.get(c["verdict"]),
                 judge=f"llm:{provider}", agreement=c.get("agreement", "1/1"),
                 why=f"{c['why']} [{c['raw_verdict']}"
                     + (f", 투표 {c['votes']}" if len(c.get("votes", [])) > 1 else "") + "]")
    _cache_save(cache)
    return calls


def judge_dir(paths, answer_dir=None):
    out = []
    for path in paths:
        for f in sorted(pathlib.Path(path).glob("*.json")):
            run = json.loads(f.read_text(encoding="utf-8"))
            ans = load_answer(run.get("series"), answer_dir)
            if ans is None:
                print(f"SKIP {f.name}: answers/{run.get('series')}.json 없음", file=sys.stderr)
                continue
            for r in grade_run(run, ans):
                out.append(dict(file=f.name, dir=str(path), series=run.get("series"),
                                model=run.get("model"), mode=run.get("mode"),
                                rag=run.get("rag"), images=run.get("images"),
                                cause_hedge=run.get("cause_hedge"),
                                pipeline_version=run.get("pipeline_version"), **r))
    return out


def summarize(rows, by=("dir", "mode")):
    """None(무채점)은 평균에서 빼고 따로 센다 — 0 점과 구분해야 한다."""
    agg = collections.defaultdict(lambda: dict(n=0, scored=0, total=0.0, ungraded=0, wrong=0))
    for r in rows:
        k = tuple(r.get(b) for b in by)
        a = agg[k]; a["n"] += 1
        if r["score"] is None:
            a["ungraded"] += 1
        else:
            a["scored"] += 1; a["total"] += r["score"]
            a["wrong"] += r["verdict"] == "incorrect"
    return {k: dict(v, mean=(v["total"] / v["scored"] if v["scored"] else None)) for k, v in agg.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="런 JSON 을 answers/ 정답과 대조해 채점한다")
    ap.add_argument("dirs", nargs="+", help="채점할 결과 디렉터리 (예: outputs_v2 outputs_rag)")
    ap.add_argument("--by", nargs="+", default=["dir", "mode"],
                    help="요약 기준 열 (dir/model/mode/rag/images/cause_hedge/series/field)")
    ap.add_argument("--csv", help="항목별 채점 결과를 CSV 로 저장")
    ap.add_argument("--answers", default=None, help="정답 디렉터리 (기본 answers/)")
    ap.add_argument("--llm", default=None, metavar="PROVIDER",
                    help="코드가 기권한 항목과 Q2 를 LLM 으로 판정 (google/anthropic/openai/mock)")
    ap.add_argument("--judge-model", default=None, help="판정에 쓸 모델 ID")
    ap.add_argument("--llm-all", action="store_true",
                    help="코드가 판정한 항목까지 전부 LLM 으로 다시 판정 (두 방식 비교용)")
    ap.add_argument("--llm-repeat", type=int, default=1, metavar="N",
                    help="같은 답을 N 번 판정해 다수결 (심판이 스스로 뒤집는 것을 줄인다). 3 권장")
    a = ap.parse_args(argv)

    rows = judge_dir(a.dirs, a.answers)
    if not rows:
        print("채점할 런이 없습니다", file=sys.stderr); return 1

    if a.llm:
        n = judge_with_llm(rows, a.llm, a.judge_model, a.answers, a.llm_all, a.llm_repeat)
        print(f"LLM 판정: 새 호출 {n}건 (나머지는 logs/judge_cache.json 재사용)", file=sys.stderr)

    if a.csv:
        cols = ["dir", "file", "series", "model", "mode", "rag", "images", "cause_hedge",
                "pipeline_version", "field", "verdict", "score", "judge", "agreement",
                "why", "answer"]
        with open(a.csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"항목 {len(rows)}건 → {a.csv}")

    _report(rows, tuple(a.by))
    return 0


def judge_kind(row):
    """이 행을 누가 판정했나. 신뢰도가 다르므로 절대 한 평균에 섞지 않는다."""
    j = row.get("judge", "")
    return "code" if j == "code" else ("keyword" if j.startswith("keyword") else "llm")


def _block(rows, by, title, caption):
    if not rows:
        return
    head = " / ".join(by)
    print(f"\n{title}")
    print(f"  {head:36} {'평균':>6} {'채점':>5} {'무채점':>6} {'오답':>5}")
    for k, v in sorted(summarize(rows, by).items(), key=lambda x: [str(y) for y in x[0]]):
        mean = f"{v['mean']:.2f}" if v["mean"] is not None else "  -"
        print(f"  {' / '.join(str(x) for x in k):36} {mean:>6} {v['scored']:5d} "
              f"{v['ungraded']:6d} {v['wrong']:5d}")
    print(f"  {caption}")


def _report(rows, by):
    """코드 채점과 LLM 판정을 **따로** 낸다.

    섞은 평균을 내지 않는 것이 이 함수의 요점이다. 코드 채점은 재현 가능하지만
    LLM 판정은 이 저장소 실측에서 자기일관성 75%(temperature 0) 였다.
    한 숫자로 합치면 그 차이가 보이지 않는다.
    """
    groups = collections.defaultdict(list)
    for r in rows:
        groups[judge_kind(r)].append(r)

    _block(groups["code"], by, "■ 코드 채점 — 주지표 (재현 가능, API 키 불필요)",
           "answers/ 의 correct/acceptable/incorrect 집합과 대조. 같은 입력이면 항상 같은 결과.")
    _block(groups["keyword"], by, "□ Q2 키워드 잠정 점수 — 참고",
           "키워드 유무만 본다. --llm 을 주면 LLM 판정으로 대체된다.")
    _block(groups["llm"], by, "□ LLM 판정 — 참고만 (결론 근거로 쓰지 말 것)",
           "자기일관성 75%(temperature 0, 표본 12). 사람 채점과 맞춰본 적 없음. "
           "--llm-repeat 3 으로 잡음을 줄일 수 있으나 편향은 남는다.")

    print("\n평균은 correct=1.0 / acceptable=0.5 / incorrect=0.0. 무채점(루브릭 밖)은 평균에서 제외.")
    if groups["llm"] or groups["keyword"]:
        print("두 블록의 평균을 합치지 마세요. 신뢰도가 다릅니다 (README 의 채점 절 참고).")


if __name__ == "__main__":
    sys.exit(main())
