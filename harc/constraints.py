"""설계 요구(하드 제약) 판정 — 규칙이 한다.

v4 기준선(2026-10-03)에서 S4 8런 중 5런이 "AR > 10 을 만족하지 못한다" 고 **스스로 적으면서** 그 조건을
공정 창으로 골랐다. 모델은 설계 요구를 넘을 수 없는 선이 아니라 절충할 항목으로 다룬다.
'요구를 넘는가' 는 숫자 비교라서 시스템 설계(수치는 규칙, 판단 보조는 LLM)대로 규칙이 맡는다.

시리즈 파일의 `constraints`:
    [{"field": "aspect_ratio", "op": ">", "value": 10, "label": "설계 요구 AR > 10"}]

규칙은 조건마다 요구 충족 여부를 판정해 후보(eligible)와 제외(excluded)를 프롬프트에 넣고,
모델 답이 제외 조건을 고르면 위반으로 기록한다. **모델 답은 고치지 않는다** — 채점은 모델이 낸 답 그대로.

**v5.1 — 지배 관계도 규칙이 계산한다.** v5 확인(2026-10-05)에서 제약은 16런 모두 지켰지만 Flash-Lite 는
후보 목록을 통째로 냈다. 후보 안에서 다른 후보에 **모든 목표 지표에서 지는** 조건(지배당한 조건)을 가려내는 것도
숫자 비교다. 시리즈 파일의 `objectives`(논문이 내건 목표 지표)로 판정한다:
    [{"field": "aspect_ratio", "better": "higher", "label": "종횡비"}, ...]
목표 지표를 결함 모듈에서 짐작하지 않는 이유: S4 는 목표가 넷(종횡비·언더컷·측벽각·등방성)인데 등방성은 결함
모듈이 없다. 셋으로만 판정하면 E3(등방성 최선)가 지배당한 것으로 잘못 찍힌다.
"""
import operator
import re

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def evaluate(series):
    """시리즈에 제약이 없으면 None."""
    cons = series.get("constraints") or []
    if not cons:
        return None
    eligible, excluded, unknown = [], {}, []
    for c in series["conditions"]:
        reasons, missing = [], False
        for k in cons:
            v = c.get(k["field"])
            if v is None:
                missing = True
                continue
            if not OPS[k["op"]](v, k["value"]):
                reasons.append(f"{k['field']} = {v} — {k.get('label') or (k['field'] + k['op'] + str(k['value']))} 미충족")
        if reasons:
            excluded[c["id"]] = reasons
        elif missing:
            unknown.append(c["id"])
        else:
            eligible.append(c["id"])
    out = {"constraints": [k.get("label") or f"{k['field']} {k['op']} {k['value']}" for k in cons],
           "eligible": eligible, "excluded": excluded, "unknown": unknown}
    objs = series.get("objectives") or []
    if objs and len(eligible) > 1:
        out.update(pareto(series, eligible, objs))
    return out


def pareto(series, ids, objs):
    """후보 ids 사이의 지배 관계. 값이 없는 지표는 그 지표를 가진 조건끼리만 비교하지 않고 통째로 뺀다."""
    conds = {c["id"]: c for c in series["conditions"]}
    objs = [o for o in objs if all(conds[i].get(o["field"]) is not None for i in ids)]
    def better(a, b, o):
        x, y = conds[a][o["field"]], conds[b][o["field"]]
        return x > y if o["better"] == "higher" else x < y
    def dominates(a, b):
        return (all(not better(b, a, o) for o in objs) and any(better(a, b, o) for o in objs))
    dominated = {b: [a for a in ids if a != b and dominates(a, b)] for b in ids}
    dominated = {b: a for b, a in dominated.items() if a}
    table = {i: {o.get("label") or o["field"]: conds[i][o["field"]] for o in objs} for i in ids}
    return {"objectives": [f"{o.get('label') or o['field']}({'클수록' if o['better'] == 'higher' else '작을수록'} 좋음)"
                           for o in objs],
            "candidate_values": table,
            "dominated": dominated,
            "pareto_front": [i for i in ids if i not in dominated]}


def prompt_block(result):
    """프롬프트에 붙일 문단. 제약이 없으면 빈 문자열."""
    if not result:
        return ""
    import json
    return ("\n\n설계 요구 판정 (규칙이 계산 — 그대로 따르세요):\n"
            + json.dumps(result, ensure_ascii=False, indent=1)
            + "\n- excluded 의 조건은 설계 요구를 못 맞춘다. process_window 에 넣지 말고, eligible 안에서 고르세요."
              " 다른 결함과의 절충은 eligible 안에서 하세요."
            + ("\n- dominated 의 조건은 다른 후보에 모든 목표 지표에서 진다. 빼세요."
               " pareto_front 안에서 목표에 가장 맞는 조건을 골라 process_window 로 내고,"
               " 후보를 통째로 내지 마세요. 무엇을 감수하는지 tradeoffs 에 적으세요."
               if result.get("pareto_front") else ""))


def chosen_ids(text, series):
    """답 문자열에서 조건 id 를 뽑는다 (id 만 본다 — 라벨 매칭은 채점기의 일)."""
    if text is None:
        return []
    t = text if isinstance(text, str) else str(text)
    out = []
    for c in series["conditions"]:
        cid = c["id"]
        tail = r"(?![0-9])" if cid[-1].isdigit() else r"(?![A-Za-z0-9])"
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(cid)}{tail}", t):
            out.append(cid)
    return out


def check(answer_window, series, result):
    """모델이 고른 공정 창이 제약을 어겼는가."""
    if not result:
        return None
    ids = chosen_ids(answer_window, series)
    return {"chosen": ids, "violations": [i for i in ids if i in result["excluded"]]}
