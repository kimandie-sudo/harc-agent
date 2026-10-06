"""기제 실험 v2 실행기 (DESIGN_기제실험.md 3절).

single 모드, 다섯 팔을 **블록 안에서 무작위 순서로 연달아** 돌린다.
블록 = (시리즈, 모델, 반복). 팔 순서와 블록 순서는 고정 시드로 정해져 재현된다.

- **반복 단위로 먼저 돈다** (rep 1 의 모든 셀 → rep 2 …). 할당량이 중간에 끊겨도
  완성된 반복이 셀마다 고르게 남는다
- **재개 가능**: 이미 있는 (블록, 팔) 파일은 건너뛴다
- **연속 실패 3회면 멈춘다.** 2026-09-24 에 실패가 쌓이는데도 계속 돌려 26B 할당량을
  645/500 까지 태웠다. 실패한 팔은 파일을 남기지 않고, 분석기가 미완 블록을 통째로 버린다
- 시작할 때 `_schedule.json` 을 쓴다. 이미 있는데 **설계(시드·시리즈·모델·반복)가 다르면
  돌지 않는다** — 다른 설계의 런이 한 디렉터리에 섞이면 블록 짝이 깨진다

씀:
    python tools/mech_run.py --dry-run                 # 일정만 출력
    python tools/mech_run.py                           # 본 실행 (google, 두 모델, 10반복)
    python tools/mech_run.py --provider mock --reps 1 --outdir /tmp/mech   # 배관 점검
"""
import argparse
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DESIGN = "v2"
SEED = 20260925
SERIES = ("S1", "S2", "S3", "S4")
MODELS = ("gemini-3.5-flash-lite", "gemma-4-26b-a4b-it")
REPS = 10
MAX_CONSECUTIVE_FAIL = 3

# 모든 팔에 공통. 팔마다 다른 것은 ARMS 에만 둔다.
COMMON = dict(mode="single", retriever="bm25", anonymize_chunks=True,
              temperature=0.0, rule_hints=True)
ARMS = {
    "off": dict(use_rag=False),
    "off2": dict(use_rag=False),                          # off′ — 잡음 바닥
    "rag": dict(use_rag=True, chunk_source="rag"),
    "placebo": dict(use_rag=True, chunk_source="placebo"),
    "planted": dict(use_rag=True, chunk_source="planted"),
}


def slug(model):
    return re.sub(r"[^a-z0-9]+", "-", str(model).lower()).strip("-") or "default"


def block_id(series, model, rep):
    return f"{series}_{slug(model)}_r{rep:02d}"


def schedule(series=SERIES, models=MODELS, reps=REPS, seed=SEED):
    """블록 목록. 반복이 바깥, 셀 순서는 반복마다 섞고, 팔 순서는 블록마다 섞는다."""
    out = []
    cells = [(s, m) for s in series for m in models]
    for rep in range(1, reps + 1):
        order = cells[:]
        random.Random(f"{seed}:rep{rep}").shuffle(order)
        for s, m in order:
            b = block_id(s, m, rep)
            arms = list(ARMS)
            random.Random(f"{seed}:{b}").shuffle(arms)
            out.append({"block": b, "series": s, "model": m, "rep": rep, "arms": arms})
    return out


def design_of(series, models, reps, seed):
    return {"design": DESIGN, "seed": seed, "series": list(series), "models": list(models),
            "reps": reps, "arms": list(ARMS), "common": COMMON}


def fname(block, arm):
    return f"{block}__{arm}.json"


def default_runner(provider):
    from harc.pipeline import run
    from harc.llm import DEFAULT_MODELS

    def _run(series, model, arm_kw):
        res = run(series, provider=provider, model=None if provider == "mock" else model,
                  base_dir=str(ROOT / "data"), **COMMON, **arm_kw)
        res["model"] = model if provider != "mock" else DEFAULT_MODELS.get("mock")
        return res
    return _run


def run_schedule(sched, outdir, runner, seed=SEED, max_fail=MAX_CONSECUTIVE_FAIL, log=print):
    """일정대로 돈다. 반환: {"done", "skipped", "failed", "stopped"}."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stats = {"done": 0, "skipped": 0, "failed": 0, "stopped": False}
    streak = 0
    series_cache = {}
    for blk in sched:
        for order, arm in enumerate(blk["arms"]):
            p = outdir / fname(blk["block"], arm)
            if p.exists():
                stats["skipped"] += 1
                continue
            s = blk["series"]
            if s not in series_cache:
                series_cache[s] = json.loads(
                    (ROOT / "data" / "series" / f"{s}.json").read_text(encoding="utf-8"))
            try:
                res = runner(series_cache[s], blk["model"], ARMS[arm])
            except Exception as e:                     # noqa: BLE001 — 어떤 실패든 센다
                stats["failed"] += 1
                streak += 1
                log(f"FAIL {blk['block']} {arm}: {str(e)[:200]}")
                if streak >= max_fail:
                    log(f"연속 실패 {streak}회 — 멈춘다. 원인부터 볼 것 (할당량? 형식 오류?)")
                    stats["stopped"] = True
                    return stats
                continue
            streak = 0
            res["mech"] = {"design": DESIGN, "seed": seed, "block": blk["block"], "arm": arm,
                           "order": order, "series": s, "rep": blk["rep"]}
            p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            stats["done"] += 1
            log(f"OK   {blk['block']} {arm} ({order + 1}/5)")
    return stats


def write_or_check_schedule(outdir, design, sched):
    """_schedule.json 을 쓰거나, 이미 있으면 같은 설계인지 확인한다."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "_schedule.json"
    doc = {"design": design, "blocks": sched}
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old.get("design") != design:
            raise SystemExit(f"{p} 의 설계가 다르다 — 다른 디렉터리를 쓰거나 기존 일정을 확인할 것\n"
                             f"  기존: {old.get('design')}\n  지금: {design}")
        return
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", default="google", choices=["google", "mock"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--series", nargs="+", default=list(SERIES))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default="outputs_mech")
    ap.add_argument("--dry-run", action="store_true", help="일정만 출력하고 돌지 않는다")
    a = ap.parse_args(argv)

    sched = schedule(a.series, a.models, a.reps, a.seed)
    design = design_of(a.series, a.models, a.reps, a.seed)
    n = len(sched) * len(ARMS)
    print(f"설계 {DESIGN} · 시드 {a.seed} · 블록 {len(sched)} · 런 {n}")
    if a.dry_run:
        for b in sched:
            print(f"  {b['block']:40s} {' → '.join(b['arms'])}")
        return 0
    outdir = ROOT / a.outdir if not pathlib.Path(a.outdir).is_absolute() else pathlib.Path(a.outdir)
    write_or_check_schedule(outdir, design, sched)
    stats = run_schedule(sched, outdir, default_runner(a.provider), seed=a.seed)
    print(f"완료 {stats['done']} · 건너뜀 {stats['skipped']} · 실패 {stats['failed']}"
          + (" · **중단**" if stats["stopped"] else ""))
    return 2 if stats["stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
