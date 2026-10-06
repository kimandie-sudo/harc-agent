"""채움률 — 주지표 항목이 실제로 답해졌는지 센다.

**모드마다 답이 들어가는 자리가 다르다.** multi 는 `run["aggregate"]`, single 은
`run["single"]["q1"]` 아래다. 한쪽만 보면 나머지 한 모드가 통째로 '키 없음' 으로 잡힌다
(2026-09-25 에 그렇게 잘못 셌다).

배치끼리 비교할 때는 **모델을 맞춰야 한다** — 배치마다 모델 구성이 다르면
채움률 차이가 모델 차이인지 수정 효과인지 갈리지 않는다.

씀: python tools/fill_rate.py outputs_retriever4 outputs_retriever6
    python tools/fill_rate.py outputs_retriever4 --model gemini-3.5-flash-lite
"""
import collections
import json
import pathlib
import sys

FIELDS = ("process_window", "primary_issue", "onset_condition")


def field(run, name):
    if run.get("mode") == "single":
        return (run.get("single") or {}).get("q1", {}).get(name, "<키없음>")
    return (run.get("aggregate") or {}).get(name, "<키없음>")


def kind(v):
    """`null` 과 '키 없음' 을 구분한다 — 전자는 프롬프트가 허용한 답이고 후자는 형식 오류다."""
    if v == "<키없음>":
        return "키없음"
    if v is None:
        return "null"
    if isinstance(v, str) and not v.strip():
        return "빈문자"
    if v == []:
        return "빈리스트"
    return "값있음"


def report(d, name="process_window", model=None):
    files = sorted(pathlib.Path(d).glob("*.json"))
    if not files:
        print(f"{d}: 런 없음")
        return
    by_mode = collections.defaultdict(collections.Counter)
    by_cell = collections.defaultdict(collections.Counter)
    kept = 0
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        if model and r.get("model") != model:
            continue
        kept += 1
        k = kind(field(r, name))
        by_mode[r["mode"]][k] += 1
        by_cell[(r["series"], r["mode"])][k] += 1
    tot = kept
    if not tot:
        print(f"\n== {d}  {name}  (model={model}) 해당 런 없음")
        return
    filled = sum(c["값있음"] for c in by_mode.values())
    tag = f"  [{model}]" if model else ""
    print(f"\n== {d}{tag}  {name}  {tot}런 → 채움 {filled}/{tot} = {filled / tot:.0%}")
    for m in sorted(by_mode):
        c = by_mode[m]
        rest = " ".join(f"{k}={v}" for k, v in sorted(c.items()) if k != "값있음")
        print(f"   {m:7s} {c['값있음']:3d}/{sum(c.values()):<3d} {rest}")
    for key in sorted(by_cell):
        c = by_cell[key]
        rest = " ".join(f"{k}={v}" for k, v in sorted(c.items()) if k != "값있음")
        print(f"     {key[0]} {key[1]:7s} {c['값있음']}/{sum(c.values())}  {rest}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["outputs_retriever4"]
    model = None
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        args = args[:i] + args[i + 2:]
    fields = FIELDS if "--all-fields" in args else ("process_window",)
    for d in [a for a in args if not a.startswith("--")]:
        for f in fields:
            report(d, f, model)
