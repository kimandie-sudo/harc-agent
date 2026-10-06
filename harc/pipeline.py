"""[1]→[5] 파이프라인. mode: multi(에이전트 분리) / single(한 번에). rag, images 는 실험 조건 플래그."""
import json, re, sys, uuid, datetime, pathlib
from concurrent.futures import ThreadPoolExecutor
from . import PIPELINE_VERSION
from .checker import check
from .defects.registry import ALL
from .llm import call_json, MAX_PARALLEL
from . import constraints as hard
from . import rag

J = lambda o: json.dumps(o, ensure_ascii=False, indent=1)
BASE = """당신은 반도체 식각(고종횡비 콘택) 공정개발 엔지니어를 돕는 분석 에이전트입니다.
입력에 있는 숫자와 판정만 근거로 삼고, 새 숫자를 만들지 마세요. 판정 불가 항목은 추측하지 마세요.
반드시 JSON 객체 하나만 출력하세요."""

AGG_SYSTEM = BASE + """
역할: 결함별 분석 결과를 취합해 이 공정 시리즈에서 무엇이 가장 중대한지 판단합니다 (원인은 다루지 않음).
공정 창: 이 시리즈에서 쓸 조건 id 를 **반드시** 지목하세요. 모든 조건에 문제가 있어도 가장 나은 조건을 고르고, 무엇을 감수하는지 tradeoffs 에 적으세요.
시작 조건: 가장 중대한 문제가 처음 나타나거나 뚜렷해지는 조건 id 를 지목하세요.
서로 반대 방향 문제(예: 한쪽 끝은 보잉, 다른 끝은 좁아짐)가 있으면 트레이드오프로 명시하세요.
출력 키: {"ranking": [{"defect": , "severity": 0-3, "why": }], "primary_issue": , "onset_condition": "조건 id",
"process_window": "조건 id 목록", "tradeoffs": [문장], "confidence": "high/medium/low"}"""

# cause 프롬프트의 인용 지시. hedge=True 가 v2 까지의 판이다.
# outputs_rag/ 32런에서 multi(=cause)는 rag-on 8런 모두 인용 0건, single 은 8런 모두 인용이었다.
# 유력한 원인이 hedge 판의 면책 조항("청크에 없으면 basis 를 general_knowledge 로")과 basis 필드라서,
# 그것을 뺀 판(hedge=False, single 의 지시와 같은 수준)을 두고 --cause-hedge 로 고를 수 있게 했다.
_CAUSE_CITE = {
    True:  ('- 문헌 청크가 주어지면 근거로 쓴 청크의 chunk_id 를 citations 에 넣으세요.'
            ' 청크에 없는 내용은 citations 를 비우고 basis 를 "general_knowledge"로 표시하세요.'),
    False: '- 문헌 청크가 주어지면 근거로 쓴 청크의 chunk_id 를 citations 에 넣으세요.',
}
_CAUSE_KEYS = {
    True:  '"verdict": "supported/excluded/undetermined", "basis": "literature/general_knowledge", "citations": [chunk_id]}],',
    False: '"verdict": "supported/excluded/undetermined", "citations": [chunk_id]}],',
}

def cause_system(hedge=True):
    return BASE + """
역할: 중대한 문제에 대해서만 공정 기제 후보를 세우고, 관측 추세와 대조해 지지/배제합니다.
- 후보는 문제당 최대 3개. 각 후보가 '예측하는 추세'를 먼저 쓰고, 입력의 '관측 추세'와 비교해 판정하세요.
""" + _CAUSE_CITE[hedge] + """
- 바꾼 변수(swept_variable) 외의 요인(장비 이력, 챔버 상태 등)은 이 데이터로 확인 불가라고 명시하세요.
출력 키: {"hypotheses": [{"issue": , "mechanism": , "predicted_trend": , "observed_trend": ,
""" + _CAUSE_KEYS[hedge] + """
"cannot_assess": [문장]}"""

NEXT_SYSTEM = BASE + """
역할: 앞 단계 결과만 보고 다음 DOE 스플릿을 제안합니다.
출력 키: {"recommendation": {"variable": , "direction": , "suggested_levels": , "expected_effect": , "risk": },
"alternatives": [{"variable": , "direction": , "why": }], "measure_next": [다음 실험에서 꼭 재야 할 값], "confidence": "high/medium/low"}"""

# --- 규칙 출력의 '방향 힌트' 제거 (--rule-hints off) ---
# v2 는 방향·최악/최선 조건·개선/악화 판정을 코드로 계산해 프롬프트에 넣어 준다.
# 그런데 채점 항목(onset=worst_condition, process_window=best_condition)이 바로 그 값들이라,
# 모델은 받은 값을 옮기기만 해도 맞는다 — 문헌이 기여할 여지가 없다.
# 이 플래그는 힌트를 빼서 모델이 스스로 방향을 판단하게 만든다. RAG 가 기여할 자리를 만드는 것이 목적이고,
# 동시에 v2 의 기여도 자체를 재는 ablation 이기도 하다.
# **결과 JSON 의 rules 는 온전한 채로 둔다** — 기록이자 채점 기준이므로 건드리지 않는다.
_HINT_KEYS = ("direction", "direction_note", "summary", "best_condition", "best_value",
              "worst_condition", "worst_value", "onset", "change_vs_first_basis",
              "metric_definition")
_HINT_ROW_KEYS = ("change_vs_first", "significant_vs_first")

def strip_direction_hints(rules):
    """규칙 판정에서 방향을 알려 주는 항목만 뺀 사본. 원본은 그대로 둔다."""
    out = {}
    for name, r in rules.items():
        if not isinstance(r, dict):
            out[name] = r
            continue
        d = {k: v for k, v in r.items() if k not in _HINT_KEYS}
        if isinstance(r.get("rows"), list):
            d["rows"] = [{k: v for k, v in row.items() if k not in _HINT_ROW_KEYS}
                         if isinstance(row, dict) else row for row in r["rows"]]
        f = r.get("factorial")
        if isinstance(f, dict):                     # 2x2 주효과의 verdict·direction 도 방향 판정이다
            me = f.get("main_effects_pct")
            d["factorial"] = {k: v for k, v in f.items()
                              if k not in ("direction", "note", "main_effects_pct")}
            if isinstance(me, dict):
                d["factorial"]["main_effects_pct"] = {
                    fac: [{kk: vv for kk, vv in e.items() if kk != "verdict"}
                          if isinstance(e, dict) else e for e in lst]
                    for fac, lst in me.items()}
        out[name] = d
    return out

def defect_glossary(hints=True):
    """결함 정의와 지표 방향. multi 는 결함별 에이전트 프롬프트로 같은 내용을 받는다."""
    lines = []
    for d in ALL:
        line = d.glossary_line()
        if not hints:                               # '(클수록 심함) · 방향 …' 을 떼어낸다
            line = re.split(r"\s*\((?:클수록|작을수록) 심함\)|\s*·\s*방향", line)[0].rstrip(" ·")
        lines.append(line)
    return "\n".join(lines)

SINGLE_SYSTEM_TMPL = BASE + """
역할: 아래 공정 시리즈 데이터를 보고 한 번에 답합니다.
Q1 가장 중대한 문제와 시작 조건, Q2 기제 후보(최대 3개, 지지/배제 근거), Q3 다음 스플릿 제안.
문헌 청크가 주어지면 chunk_id 를 인용하세요.

결함 정의와 지표 방향 (규칙 판정의 direction/summary 와 같은 기준입니다):
{glossary}

{direction_rules}
- '판정 불가' 조건은 추측하지 마세요.
- process_window 에는 이 시리즈에서 쓸 조건 id 를 반드시 지목하세요. 모든 조건에 문제가 있어도 가장 나은 조건을 고르고, 무엇을 감수하는지 tradeoffs 에 적으세요.

출력 키: {"q1": {"primary_issue": , "onset_condition": , "process_window": , "tradeoffs": []},
"q2": {"hypotheses": [{"mechanism": , "predicted_trend": , "observed_trend": , "verdict": , "citations": []}]},
"q3": {"variable": , "direction": , "suggested_levels": , "risk": }}"""

_SINGLE_DIR_RULES = {
    True: ("- 좋고 나쁨의 방향은 위 '방향'과 입력의 direction/summary 를 따르세요. 값의 크고 작음을 임의로 뒤집지 마세요.\n"
           "- 조건별 change_vs_first(\"개선\"/\"악화\"/\"변화 없음\")와 worst_condition/best_condition 은 코드가 계산한 값입니다. 그대로 쓰고 다시 판단하지 마세요."),
    False: "- 어느 쪽이 나쁜 방향인지는 주어지지 않았습니다. 지표의 정의와 공정 지식으로 스스로 판단하세요.",
}

def single_system(hints=True):
    return (SINGLE_SYSTEM_TMPL.replace("{glossary}", defect_glossary(hints))
                              .replace("{direction_rules}", _SINGLE_DIR_RULES[hints]))

def _series_brief(s):
    # goal(목표·설계 요구)은 v4 부터 넣는다 — 저자 결론이 아니라 논문이 내건 목표다 (2026-10-03)
    return {k: s.get(k) for k in ("series_id", "structure", "goal", "swept_variable", "other_fixed")
            if k != "goal" or s.get(k)}

def _rag(series, query, use_rag, warnings, chunk_chars=None,
         retriever="bm25", embed_provider="google", chunk_source="rag"):
    """RAG 검색 + 안전장치. 경고는 warnings 에 쌓고 stderr 로도 알린다.

    chunk_source="placebo" 면 **검색은 그대로 돌리되** 그 결과의 단어 수에 맞춘
    무관한 청크로 바꿔 넣는다 (기제 실험용 — DESIGN_기제실험.md).
    검색을 먼저 돌리는 이유는 길이를 맞추기 위해서다.
    chunk_source="planted" 면 최하위 청크 앞에 "이 시리즈에서는 {틀린 조건}이 공정 창" 이라는
    문장을 붙인다 (양성 대조군). 익명화 없이는 쓸 수 없다 — run() 이 막는다.
    """
    if not use_rag:
        return []
    # embed/hybrid 일 때만 임베딩 색인의 최신 여부까지 본다.
    for w in rag.corpus_warnings(check_embeddings=(retriever != "bm25")):
        _warn(warnings, w)
    hits = rag.retrieve(query, k=6, exclude_papers=[series.get("source_set")],
                        retriever=retriever, embed_provider=embed_provider)
    if chunk_source == "placebo":
        for w in rag.placebo_warnings():
            _warn(warnings, w)
        hits = rag.placebo_for(hits)
    elif chunk_source == "planted":
        hits = rag.planted_for(hits, series)
    if not hits:
        _warn(warnings, f"RAG on 이지만 검색된 청크가 0개입니다 (query={query!r}, "
                        f"제외 논문={series.get('source_set')!r}). 근거 없이 답하게 되므로 "
                        f"corpus/chunks.jsonl 을 확인하세요")
    return [_pass_chunk(h, warnings, chunk_chars) for h in hits]

# 전달 상한. rag.CHUNK_WORDS(350단어) 를 넉넉히 덮는 값이라 정상 색인에서는 걸리지 않는다.
# 걸린다면 색인이 옛 판(상한 없이 만든 것)이라는 뜻이므로 경고한다 — 그때는 BM25 가 점수를 매긴
# 본문과 모델이 보는 본문이 달라져 검색이 조용히 망가진다.
CHUNK_CHAR_CAP = 3000

def _pass_chunk(hit, warnings, chunk_chars=None):
    """청크를 프롬프트로 넘긴다. chunk_chars 를 주면 **일부러** 그 길이로 자른다.

    일부러 자르는 것은 실험 조건이다 (--chunk-chars). 옛 판에서 1,200자로 잘리던 것을
    같은 배치 안에서 재현해 '절단'과 '면책 조항' 을 분리하기 위한 것이라, 이때는 경고하지 않는다.
    경고는 상한을 지정하지 않았는데도 청크가 기본 상한을 넘을 때만 낸다 — 그건 색인이 옛 판이라는 뜻이다.
    """
    text = hit["text"]
    if chunk_chars:                       # 실험 조건으로 지정한 절단
        text = text[:chunk_chars]
    elif len(text) > CHUNK_CHAR_CAP:
        _warn(warnings, f"청크 {hit['chunk_id']} 가 {len(text)}자로 전달 상한({CHUNK_CHAR_CAP})을 넘어 "
                        f"잘립니다 — 색인이 옛 판입니다. `python -m harc.rag build` 로 다시 만드세요")
        text = text[:CHUNK_CHAR_CAP]
    return {"chunk_id": hit["chunk_id"], "text": text}

def _alias(chunks, out):
    """프롬프트용으로 chunk_id 를 D1…Dn 으로 바꾼다. 대응표는 out["chunk_alias"] 에 남긴다.

    chunk_id 는 `wachter2018#3` 처럼 논문 이름을 담고 있어, 위약(XAI 논문)이나 심은 청크가
    티가 난다. 기제 실험에서는 **모든 팔**을 익명화해 팔끼리 프롬프트 모양을 같게 한다.
    모델의 인용(D3 등)은 cite_eval 이 이 대응표로 되돌려 채점한다.
    """
    alias = {f"D{i + 1}": c["chunk_id"] for i, c in enumerate(chunks)}
    out["chunk_alias"] = alias
    return [{"chunk_id": a, "text": c["text"]} for a, c in zip(alias, chunks)]

def _warn(warnings, msg):
    if msg not in warnings:
        warnings.append(msg)
        print("WARN:", msg, file=sys.stderr)

def run(series, provider="mock", model=None, mode="multi", use_rag=False, use_images=False,
        base_dir=".", cause_hedge=True, chunk_chars=None, temperature=0.0,
        rule_hints=True, retriever="bm25", chunk_source="rag", anonymize_chunks=False,
        max_parallel=None):
    run_id = datetime.datetime.now().strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    kw = dict(provider=provider, model=model, run_id=run_id, temperature=temperature)
    out = {"run_id": run_id, "pipeline_version": PIPELINE_VERSION,
           "series": series["series_id"], "provider": provider, "model": model,
           "mode": mode, "rag": use_rag, "images": use_images,
           "cause_hedge": cause_hedge, "chunk_chars": chunk_chars,
           "temperature": temperature, "rule_hints": rule_hints,
           "retriever": retriever, "chunk_source": chunk_source,
           "anonymize_chunks": anonymize_chunks}
    if chunk_source == "planted":
        if not anonymize_chunks:
            raise ValueError("chunk_source='planted' 는 anonymize_chunks=True 와 함께만 쓴다 — "
                             "심은 청크의 chunk_id 가 프롬프트에 그대로 드러난다")
        if use_rag:
            out["planted"] = rag.planted_spec(series)
    warnings = []

    # [1] 데이터 점검
    chk = check(series, provider, model, use_images, run_id, base_dir, temperature)
    out["check"] = chk

    # [2] 결함별 규칙 (코드)
    rules = {d.name: d.rule(series, chk["judgeable"][d.name]) for d in ALL}
    out["rules"] = rules                      # 기록·채점용: 항상 온전한 판정
    shown = rules if rule_hints else strip_direction_hints(rules)   # 프롬프트용
    imgs = [pathlib.Path(base_dir) / c["image"] for c in series["conditions"]
            if use_images and c.get("image") and (pathlib.Path(base_dir) / c["image"]).exists()]

    # 설계 요구(하드 제약)는 규칙이 판정한다 — v5 (harc/constraints.py)
    cons = hard.evaluate(series)
    if cons:
        out["constraints"] = cons

    if mode == "single":
        chunks = _rag(series, f"{series['swept_variable']} etch profile bowing taper mechanism",
                      use_rag, warnings, chunk_chars, retriever,
                      "mock" if provider == "mock" else "google", chunk_source)
        shown_chunks = _alias(chunks, out) if (anonymize_chunks and chunks) else chunks
        user = ("시리즈:\n" + J(_series_brief(series)) + "\n\n데이터 점검:\n" + J(chk["judgeable"])
                + "\n\n규칙 판정:\n" + J(shown) + hard.prompt_block(cons)
                + ("\n\n문헌 청크:\n" + J(shown_chunks) if chunks else ""))
        out["single"] = call_json("single", system=single_system(rule_hints), user=user,
                                  required=["q1", "q2", "q3"], images=imgs, **kw)
        q1 = out["single"].get("q1") if isinstance(out["single"], dict) else None
        if cons:
            out["constraint_check"] = hard.check(
                q1.get("process_window") if isinstance(q1, dict) else q1, series, cons)
        out["rag_chunks"] = chunks
        out["rag_warning"] = warnings
        return out

    # [2] 결함별 LLM 해석 (병렬)
    def one(d):
        return d.name, call_json(f"defect:{d.name}", system=d.system_prompt(rule_hints),
                                 user=d.user_prompt(series, shown[d.name]),
                                 required=d.REQUIRED_KEYS, images=imgs, **kw)
    # 동시 호출 수는 결과에 영향이 없다 — 결함 에이전트끼리 출력을 주고받지 않는다 (llm.MAX_PARALLEL 참고)
    workers = max_parallel or MAX_PARALLEL.get(provider) or len(ALL)
    out["max_parallel"] = workers
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out["defects"] = dict(ex.map(one, ALL))

    # [3] 취합·경중
    out["aggregate"] = call_json("aggregate", system=AGG_SYSTEM,
        user="시리즈:\n" + J(_series_brief(series)) + "\n\n결함별 결과:\n" + J(out["defects"])
             + hard.prompt_block(cons),
        required=["ranking", "primary_issue", "onset_condition", "process_window", "tradeoffs"],
        # v2 는 "조건 id 목록 또는 null" 이라 process_window 를 비워도 됐다 (v2 multi 276런 중 76% 가 null).
        # v3 은 반드시 지목하게 한다. onset 은 모든 조건이 같은 상태면 정할 수 없으므로 비어도 된다.
        nullable=["onset_condition"], **kw)
    if cons:
        out["constraint_check"] = hard.check(out["aggregate"].get("process_window"), series, cons)

    # [4] 원인 후보 (+RAG)
    q = f"{series['swept_variable']} {out['aggregate'].get('primary_issue')} mechanism sidewall passivation"
    chunks = _rag(series, q, use_rag, warnings, chunk_chars, retriever,
                  "mock" if provider == "mock" else "google", chunk_source)
    out["rag_chunks"] = chunks
    out["rag_warning"] = warnings
    out["cause"] = call_json("cause", system=cause_system(cause_hedge),
        user="시리즈:\n" + J(_series_brief(series)) + "\n\n중대성 판단:\n" + J(out["aggregate"])
             + "\n\n규칙 판정(추세 확인용):\n" + J(shown)
             + ("\n\n문헌 청크:\n" + J(_alias(chunks, out) if anonymize_chunks else chunks)
                if chunks else "\n\n(문헌 청크 없음)"),
        required=["hypotheses", "cannot_assess"], **kw)

    # [5] 다음 조치
    out["next"] = call_json("next", system=NEXT_SYSTEM,
        user="시리즈:\n" + J(_series_brief(series)) + "\n\n중대성:\n" + J(out["aggregate"])
             + "\n\n원인 후보:\n" + J(out["cause"]),
        required=["recommendation", "alternatives", "measure_next"], **kw)
    return out
