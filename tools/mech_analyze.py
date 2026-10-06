"""기제 실험 v2 분석기 (DESIGN_기제실험.md 4절). **사전 등록 대상이다 — 실행 후 고치지 않는다.**

주지표 셋, 모두 q1.process_window · single · 조건 집합 기준 (루브릭 점수를 거치지 않는다):

  P1 (관문)  planted 채택률 − off2 의 표적 채택률        블록 내 부호 뒤집기 순열검정, 단측
  P2         rag 움직임률   − off2 움직임률              〃, 양측
  P3         placebo 움직임률 − off2 움직임률            〃, 양측

  움직임 = 그 팔의 조건 집합 ≠ 같은 블록 off 의 조건 집합 (빈 집합도 하나의 값으로 본다)
  채택   = 표적 ∈ 조건 집합 이고 정답 조건 ∩ 조건 집합 = ∅   (구간 답은 정답을 포함해 걸러진다)
  Bonferroni: 각 α = 0.05 / 3

미완 블록(다섯 팔이 다 없는 블록)은 통째로 버린다.

P1 이 실패했을 때 두 경우를 가른다 (single 프롬프트가 규칙 판정의 best_condition 을 주면서
"코드 값을 그대로 쓰라" 고 지시하므로, 안 따른 것이 '안 읽음' 인지 '읽었지만 규칙을 따름' 인지
모른다):
  R  최하위 자리(D{n}) 인용률: planted − rag   블록 내 순열검정, 단측, α = 0.05
     같은 자리, 같은 원문 — 다른 것은 심은 문장 하나뿐이다
  P1 실패 + R 유의   → "0R" 읽지만 공정 창은 규칙 판정을 따른다
  P1 실패 + R 비유의 → "0"  읽은 흔적이 없다

부지표(탐색적, 판정에 안 씀): 루브릭 점수 차 · onset 움직임 · 외부 조건 유입 · 입력 토큰 길이.

씀: python tools/mech_analyze.py outputs_mech [--json out.json]
"""
import argparse
import collections
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harc.judge import run_answers, parse_conditions, grade_conditions  # noqa: E402

ARMS = ("off", "off2", "rag", "placebo", "planted")
ALPHA = 0.05 / 3
ALPHA_R = 0.05            # P1 실패 시에만 보는 읽음 지표. 주지표가 아니라 보정하지 않는다
N_PERM = 10_000
PERM_SEED = 20260925
SCORE = {"correct": 1.0, "acceptable": 0.5, "incorrect": 0.0}
TARGETS = ROOT / "data" / "planted.json"


# ── 불러오기 ────────────────────────────────────────────────────────────────
def load(outdir):
    runs = []
    for f in sorted(pathlib.Path(outdir).glob("*.json")):
        if f.name.startswith("_"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        if (r.get("mech") or {}).get("arm") in ARMS:
            runs.append(r)
    return runs


def complete_blocks(runs):
    """{block: {arm: run}} 중 다섯 팔이 다 있는 것만. 두 번째 값은 버린 블록 수."""
    by = collections.defaultdict(dict)
    for r in runs:
        by[r["mech"]["block"]][r["mech"]["arm"]] = r
    full = {b: arms for b, arms in by.items() if set(arms) == set(ARMS)}
    return full, len(by) - len(full)


# ── 측정 ────────────────────────────────────────────────────────────────────
_ANS = {}


def answer_key(series):
    if series not in _ANS:
        _ANS[series] = json.loads((ROOT / "answers" / f"{series}.json").read_text(encoding="utf-8"))
    return _ANS[series]


def cond_set(run, field="q1.process_window"):
    a = answer_key(run["series"])
    return frozenset(parse_conditions(run_answers(run)[field], a["conditions"]))


def moved(arm_run, off_run, field="q1.process_window"):
    return int(cond_set(arm_run, field) != cond_set(off_run, field))


def adopted(run, target):
    got = cond_set(run)
    correct = set(answer_key(run["series"])["q1"]["process_window"].get("correct", []))
    return int(target in got and not (got & correct))


def score(run, field="q1.process_window"):
    a = answer_key(run["series"])
    v, _ = grade_conditions(run_answers(run)[field], a["q1"][field.split(".")[1]], a["conditions"])
    return SCORE.get(v)


# ── 검정 ────────────────────────────────────────────────────────────────────
def sign_flip_test(diffs, alternative="two-sided", n_perm=N_PERM, seed=PERM_SEED):
    """블록 내 순열검정. 한 블록 안에서 두 팔의 이름표를 바꾸면 차이의 부호가 뒤집힌다.
    diffs: 블록마다 (팔 A − 팔 B). 반환 (평균차, p)."""
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0
    obs = sum(diffs) / n
    rng = random.Random(seed)
    hit = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if alternative == "greater":
            hit += s >= obs - 1e-12
        else:
            hit += abs(s) >= abs(obs) - 1e-12
    return obs, (hit + 1) / (n_perm + 1)


def cited_last(run):
    """최하위 청크(익명 id D{n})를 q2 에서 인용했는가. planted 팔에서는 그 자리가 심은 청크다."""
    alias = run.get("chunk_alias") or {}
    if not alias:
        return 0
    last = f"D{len(alias)}"
    q2 = (run.get("single") or {}).get("q2")          # 모델이 스키마를 벗어나도 죽지 않는다
    hyps = q2.get("hypotheses") if isinstance(q2, dict) else None
    if not isinstance(hyps, list):
        return 0
    for h in hyps:
        c = h.get("citations") if isinstance(h, dict) else None
        if isinstance(c, str):
            c = [c]
        if isinstance(c, list) and last in c:
            return 1
    return 0


def decide(p1, p2, p3, pr=1.0):
    """DESIGN 4절의 판정 규칙을 그대로 옮긴 것. pr 은 읽음 지표 R 의 p (P1 실패 때만 쓴다)."""
    s1, s2, s3 = (p < ALPHA for p in (p1, p2, p3))
    if not s1:
        if pr < ALPHA_R:
            return "0R", ("심은 청크를 읽고 인용까지 하지만 공정 창은 바꾸지 않는다 — "
                          "규칙 판정(best_condition)이 청크를 이긴다. B·C 는 이 설정에서 검정되지 않는다")
        return "0", "심어도 따라가지 않고 읽은 흔적도 없다 — q1 에 청크 내용이 쓰이지 않는다. B·C 는 성립할 수 없다"
    if not s2 and not s3:
        return "A", "모델은 청크를 읽지만(P1) 실제 문헌도 길이도 답을 움직이지 않는다 — 중복"
    if s2 and not s3:
        return "C", "내용이 움직인다 — 실제 문헌만 잡음보다 답을 더 움직인다"
    if s2 and s3:
        return "B", "길이가 움직인다 — 무관한 텍스트도 잡음보다 답을 더 움직인다 (P2·P3 크기 비교할 것)"
    return "?", "예상 밖 — 위약만 답을 움직인다. 그대로 보고한다"


# ── 부지표 ──────────────────────────────────────────────────────────────────
_UNIT = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?\s?(?:mTorr|Torr|Pa|kW|W|kV|V|sccm|°C|℃|MHz|kHz|nm|µm|um)"
                   r"(?![A-Za-z])")


def unit_tokens(text):
    return {re.sub(r"\s+", "", m.group(0)).lower() for m in _UNIT.finditer(str(text or ""))}


_SERIES_TOK = {}


def intrusion(run):
    """q1 답에 **청크에는 있고 시리즈 데이터에는 없는** 수치+단위 토큰이 있는가."""
    s = run["series"]
    if s not in _SERIES_TOK:
        _SERIES_TOK[s] = unit_tokens((ROOT / "data" / "series" / f"{s}.json").read_text(encoding="utf-8"))
    chunk_tok = set().union(*(unit_tokens(c.get("text")) for c in (run.get("rag_chunks") or [])), set())
    ans_tok = unit_tokens(json.dumps((run.get("single") or {}).get("q1"), ensure_ascii=False))
    return int(bool((ans_tok & chunk_tok) - _SERIES_TOK[s]))


def input_tokens(run_ids, log=ROOT / "logs" / "calls.jsonl"):
    """run_id → 성공한 single 호출의 입력 토큰."""
    out = {}
    if not pathlib.Path(log).exists():
        return out
    for line in open(log, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("run_id") in run_ids and r.get("ok") is True and r.get("agent") == "single":
            u = r.get("usage") or {}
            if isinstance(u, dict) and u.get("in"):
                out[r["run_id"]] = u["in"]
    return out


# ── 보고 ────────────────────────────────────────────────────────────────────
def analyze(outdir):
    runs = load(outdir)
    blocks, dropped = complete_blocks(runs)
    targets = json.loads(TARGETS.read_text(encoding="utf-8"))
    res = {"blocks": len(blocks), "dropped_incomplete": dropped, "runs": len(runs)}
    if not blocks:
        return res

    B = sorted(blocks)
    mv = {arm: [moved(blocks[b][arm], blocks[b]["off"]) for b in B] for arm in ("off2", "rag", "placebo", "planted")}
    ad = {arm: [adopted(blocks[b][arm], targets[blocks[b][arm]["series"]]) for b in B] for arm in ARMS}
    res["movement_rate"] = {a: sum(v) / len(v) for a, v in mv.items()}
    res["adoption_rate"] = {a: sum(v) / len(v) for a, v in ad.items()}

    d1, p1 = sign_flip_test([x - y for x, y in zip(ad["planted"], ad["off2"])], "greater")
    d2, p2 = sign_flip_test([x - y for x, y in zip(mv["rag"], mv["off2"])])
    d3, p3 = sign_flip_test([x - y for x, y in zip(mv["placebo"], mv["off2"])])
    res["primary"] = {"P1": {"diff": d1, "p": p1}, "P2": {"diff": d2, "p": p2}, "P3": {"diff": d3, "p": p3},
                      "alpha_each": ALPHA}
    cl = {arm: [cited_last(blocks[b][arm]) for b in B] for arm in ("rag", "planted")}
    dr, pr = sign_flip_test([x - y for x, y in zip(cl["planted"], cl["rag"])], "greater")
    res["reading"] = {"R": {"diff": dr, "p": pr, "alpha": ALPHA_R},
                      "cited_last_rate": {a: sum(v) / len(v) for a, v in cl.items()}}
    res["verdict"], res["verdict_text"] = decide(p1, p2, p3, pr)

    # 부지표 — 탐색적
    sec = {}
    by_series = collections.defaultdict(lambda: collections.defaultdict(list))
    for b in B:
        off = score(blocks[b]["off"])
        for arm in ("off2", "rag", "placebo", "planted"):
            s = score(blocks[b][arm])
            if off is not None and s is not None:
                by_series[blocks[b]["off"]["series"]][arm].append(s - off)
    sec["score_diff_vs_off"] = {s: {a: (sum(v) / len(v), len(v)) for a, v in d.items()}
                                for s, d in sorted(by_series.items())}
    sec["onset_movement_rate"] = {
        arm: sum(moved(blocks[b][arm], blocks[b]["off"], "q1.onset_condition") for b in B) / len(B)
        for arm in ("off2", "rag", "placebo", "planted")}
    sec["planted_adoption_by_series"] = {
        s: sum(ad["planted"][i] for i, b in enumerate(B) if blocks[b]["off"]["series"] == s)
        / max(1, sum(1 for b in B if blocks[b]["off"]["series"] == s))
        for s in sorted({blocks[b]["off"]["series"] for b in B})}
    sec["intrusion_rate"] = {arm: sum(intrusion(blocks[b][arm]) for b in B) / len(B)
                             for arm in ("rag", "placebo", "planted")}
    ids = {blocks[b][a]["run_id"]: a for b in B for a in ("rag", "placebo", "planted")}
    tok = input_tokens(set(ids))
    per = collections.defaultdict(list)
    for rid, n in tok.items():
        per[ids[rid]].append(n)
    mean = {a: sum(v) / len(v) for a, v in per.items() if v}
    sec["input_tokens_mean"] = mean
    if mean.get("rag"):
        sec["input_tokens_ratio_vs_rag"] = {a: mean[a] / mean["rag"] for a in mean}
        sec["length_matched_within_5pct"] = all(abs(r - 1) <= 0.05 for r in sec["input_tokens_ratio_vs_rag"].values())
    res["secondary"] = sec
    return res


def report(res):
    print(f"블록 {res['blocks']} (미완으로 버림 {res['dropped_incomplete']}) · 런 {res['runs']}")
    if not res["blocks"]:
        print("분석할 완성 블록이 없다.")
        return
    print("\n움직임률 (off 대비):", {k: f"{v:.0%}" for k, v in res["movement_rate"].items()})
    print("표적 채택률:        ", {k: f"{v:.0%}" for k, v in res["adoption_rate"].items()})
    print(f"\n주지표 (각 α = {ALPHA:.4f})")
    for k, lab in (("P1", "planted 채택 − off2 (단측)"), ("P2", "rag 움직임 − off2"), ("P3", "placebo 움직임 − off2")):
        x = res["primary"][k]
        print(f"  {k} {lab:26s} {x['diff']:+.3f}  p = {x['p']:.4f}  {'유의' if x['p'] < ALPHA else '—'}")
    r = res["reading"]
    print(f"  R  최하위 자리 인용 planted − rag (단측)  {r['R']['diff']:+.3f}  p = {r['R']['p']:.4f}"
          f"  (P1 실패 때만 판정에 쓴다) · 인용률 {r['cited_last_rate']}")
    print(f"\n판정: {res['verdict']} — {res['verdict_text']}")
    sec = res.get("secondary", {})
    print("\n부지표 (탐색적 — 판정에 쓰지 않는다)")
    for k, v in sec.items():
        print(f"  {k}: {v}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir", nargs="?", default="outputs_mech")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = analyze(a.outdir)
    report(res)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
