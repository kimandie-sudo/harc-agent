"""결함 모듈 공통 틀. 결함 하나 = rule()(코드) + 해석 프롬프트(LLM)."""
import json

THRESHOLDS = {  # 잠정값 — 전문가 확인 후 수정
    "bow_index": 0.05,        # 최대CD/상단CD - 1 > 5% → 보잉
    "neck_ratio": 0.95,       # 넥CD/상단CD < 0.95 → 네킹
    "taper_change": 0.08,     # 테이퍼 지표 조건 간 변화 > 8%
    "cd_change": 0.03,        # CD 변화 > 3%
    "sidewall_angle_deg": 0.5,  # 측벽각 변화 > 0.5도 (각도는 절대 차로 잰다)
    "mask_etch_rate_nm_min": 5,  # 마스크 식각률 변화 > 5 nm/min (0·음수를 지나므로 절대 차)
    "depth_change": 0.05,     # 식각 깊이 변화 > 5%
}

# 지표의 "좋고 나쁨" 방향. 모델이 값의 크고 작음을 개선/악화로 뒤집어 읽는 것을 막는다.
HIGHER_IS_WORSE = "higher_is_worse"
LOWER_IS_WORSE = "lower_is_worse"
TRUE_IS_WORSE = "true_is_worse"          # 불리언 지표 (예: 식각 중단)

BETTER, WORSE, SAME = "개선", "악화", "변화 없음"


def trend(values):
    """스윕 순서대로 값의 추세. None 은 건너뜀."""
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return "판단 불가(값 부족)"
    d = [b - a for a, b in zip(v, v[1:])]
    if all(x > 0 for x in d): return "단조 증가"
    if all(x < 0 for x in d): return "단조 감소"
    if all(x >= 0 for x in d): return "비감소"
    if all(x <= 0 for x in d): return "비증가"
    return "비단조(창 또는 역전 있음)"


DESIGNS = ("sweep", "factorial", "sequence", "ofat")

def design_of(series):
    """시리즈의 설계 종류. 없으면 데이터 모양에서 추론한다 (기존 시리즈 호환).

      sweep     한 변수를 순서대로 쓸어본 것 — 추세가 의미 있다
      factorial 2x2 요인 — 주효과·상호작용을 보고, 추세는 의미 없다
      sequence  베이지안 최적화처럼 여러 입력이 동시에 변하는 시퀀스.
                **조건 순서가 추세를 뜻하지 않는다.** 첫 조건은 출발점일 뿐이다
      ofat      한 번에 한 변수씩 바꾼 스윕 여러 개 — 중심 조건을 공유하거나(S5) 장비가 다른(S6) 것.
                스윕마다는 추세가 있지만 **조건 목록 전체를 한 줄로 읽으면 두 변수가 섞인다**.
                그래서 전체 추세는 끈다
    """
    d = series.get("design")
    if d in DESIGNS:
        return d
    return "factorial" if series.get("factors") else "sweep"


def show_trend(series):
    """추세를 내도 되는 설계인가. sweep 만 참."""
    return design_of(series) == "sweep"


def _worse_when_larger(direction):
    return direction in (HIGHER_IS_WORSE, TRUE_IS_WORSE)


def _num(v):
    """불리언 지표도 같은 비교 규칙을 쓰도록 0/1 로 바꾼다."""
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return v


def verdict_vs_first(cur, ref, direction, tol=0.0, relative=True):
    """첫 조건 대비 개선/악화/변화 없음.

    tol 이상 움직였을 때만 방향을 말한다 (tol 은 relative=True 면 비율, False 면 절대 폭).
    값이 없으면 None — 모델이 '판정 불가'를 '변화 없음'으로 오해하지 않게 구분한다.
    """
    cur, ref = _num(cur), _num(ref)
    if cur is None or ref is None:
        return None
    if relative:
        if not ref:
            return None
        delta = cur / ref - 1
    else:
        delta = cur - ref
    if abs(delta) <= tol:
        return SAME
    return WORSE if (delta > 0) == _worse_when_larger(direction) else BETTER


def best_worst(pairs, direction):
    """pairs: [(조건 id, 값)]. 값이 None 인 조건은 제외. 동점이면 먼저 나온 조건.

    반환: {"best_condition", "best_value", "worst_condition", "worst_value"} (판정 가능 조건이 없으면 전부 None)
    """
    usable = [(cid, _num(v)) for cid, v in pairs if _num(v) is not None]
    if not usable:
        return {"best_condition": None, "best_value": None,
                "worst_condition": None, "worst_value": None}
    hi = max(usable, key=lambda t: t[1])
    lo = min(usable, key=lambda t: t[1])
    # max/min 은 동점일 때 먼저 나온 것을 돌려준다 (파이썬 보장)
    worst, best = (hi, lo) if _worse_when_larger(direction) else (lo, hi)
    raw = dict(pairs)
    return {"best_condition": best[0], "best_value": raw[best[0]],
            "worst_condition": worst[0], "worst_value": raw[worst[0]]}


def _fmt(v):
    if isinstance(v, bool):
        return "중단" if v else "정상"
    if isinstance(v, float) and v == int(v):
        return str(int(v)) if abs(v) >= 100 else f"{v:.1f}"
    return str(v)


def direction_block(pairs, direction, direction_note):
    """worst/best + 한 줄 요약. rule() 출력에 그대로 펼쳐 넣는다."""
    bw = best_worst(pairs, direction)
    if bw["worst_condition"] is None:
        summary = f"판정 가능한 조건 없음 ({direction_note})"
    else:
        summary = (f"최악 {bw['worst_condition']}({_fmt(bw['worst_value'])}), "
                   f"최선 {bw['best_condition']}({_fmt(bw['best_value'])}), {direction_note}")
    return dict(bw, summary=summary)


class Defect:
    name = ""
    label = ""
    # 필요한 값 조합: 이 중 한 세트라도 있으면 판정 가능
    requires = []
    definition = ""
    metric = ""              # rule() rows 에서 심각도를 대표하는 값의 키
    direction = HIGHER_IS_WORSE
    direction_note = ""      # 한 줄 요약에 붙는 짧은 설명 (예: "값이 작을수록 수직")
    metric_definition = ""   # 지표 계산식 (예: "상단CD/하단CD (클수록 심함)")

    def available(self, cond):
        for req in self.requires:
            if all(cond.get(k) is not None for k in req):
                return True, req
        return False, None

    def rule(self, series, judgeable):
        raise NotImplementedError

    def meta(self):
        """규칙 출력·프롬프트가 공유하는 지표 방향 정보."""
        return {"metric": self.metric, "direction": self.direction,
                "direction_note": self.direction_note,
                "metric_definition": self.metric_definition}

    def glossary_line(self):
        return (f"- {self.name} ({self.label}): {self.definition}\n"
                f"  지표 {self.metric} = {self.metric_definition} · 방향 {self.direction} "
                f"({self.direction_note})")

    def system_prompt(self, hints=True):
        """hints=False 면 방향(어느 쪽이 나쁜지)을 알려 주지 않는다 — --rule-hints off 용.

        지표 정의에서 '(클수록 심함)' 같은 꼬리표도 떼고, 코드가 계산한 방향 판정을
        따르라는 지시도 뺀다. 모델이 스스로 방향을 판단해야 한다.
        """
        import re as _re
        metric_def = self.metric_definition if hints else \
            _re.split(r"\s*\((?:클수록|작을수록) 심함\)|\s*—", self.metric_definition)[0].strip()
        head = (f"지표: {self.metric} = {metric_def}\n방향: {self.direction} — {self.direction_note}"
                if hints else f"지표: {self.metric} = {metric_def}")
        follow = ("- 좋고 나쁨의 방향은 위 '방향'과 입력의 direction/summary 를 따르세요. 값의 크고 작음을 임의로 뒤집지 마세요.\n"
                  "- 조건별 change_vs_first(\"개선\"/\"악화\"/\"변화 없음\")와 worst_condition/best_condition 은 코드가 계산한 값입니다. 그대로 쓰고 다시 판단하지 마세요."
                  if hints else
                  "- 어느 쪽이 나쁜 방향인지는 주어지지 않았습니다. 지표의 정의와 공정 지식으로 스스로 판단하세요.")
        return f"""당신은 반도체 식각 공정 엔지니어를 돕는 '{self.label}' 전담 분석 에이전트입니다.

정의: {self.definition}
{head}

규칙:
- 입력의 규칙 판정 결과(코드가 계산한 숫자와 판정)만 근거로 해석하세요. 새 숫자를 계산하거나 만들지 마세요.
{follow}
- '판정 불가'로 표시된 조건은 추측하지 말고 not_judgeable 에 사유와 함께 그대로 옮기세요.
- 공정 원인은 여기서 다루지 않습니다(다음 단계 담당). 현상과 심각도만 판단하세요.
- 반드시 JSON 객체 하나만 출력하세요.

출력 JSON 키:
{{"defect": "{self.name}",
 "present": true/false,
 "affected_conditions": [조건 id],
 "onset_condition": "처음 나타나는 조건 id 또는 null",
 "severity": 0~3 (0 없음, 1 경미, 2 주의, 3 중대),
 "trend_summary": "스윕 방향에 따른 변화 한두 문장",
 "evidence": ["규칙 결과에서 인용한 근거"],
 "confidence": "high/medium/low",
 "not_judgeable": [{{"condition": id, "reason": "..."}}]}}"""

    REQUIRED_KEYS = ["defect", "present", "affected_conditions", "severity",
                     "trend_summary", "evidence", "confidence", "not_judgeable"]

    def user_prompt(self, series, rule_out):
        return ("공정 시리즈 정보:\n" + json.dumps({k: series[k] for k in
                ("structure", "goal", "swept_variable", "other_fixed") if k in series}, ensure_ascii=False)
                + "\n\n규칙 판정 결과:\n" + json.dumps(rule_out, ensure_ascii=False, indent=1))
