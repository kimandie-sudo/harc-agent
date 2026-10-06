"""인용 적합성 — 모델이 **근거가 되는** 청크를 인용했는지 코드로만 잰다.

  python -m harc.cite_eval outputs_rag2 outputs_rag_t07
  python -m harc.cite_eval outputs_rag2 --by model mode --csv 인용.csv

왜 이것이 필요한가:
Q2(기제)의 옳고 그름은 LLM 심판으로만 잴 수 있는데, 이 저장소 실측에서 심판의 자기일관성이
temperature 0 에서도 75% 였다 (README 의 한계 절). 그래서 Q2 점수는 결론의 근거로 쓸 수 없다.
인용 적합성은 그 사이를 메운다 — "주장이 맞는가" 대신 **"주장과 인용이 맞물리는가"** 를 재고,
전부 코드로 계산되므로 같은 입력이면 항상 같은 값이 나온다.

재는 것 (인용 하나마다):
  in_retrieved  인용한 chunk_id 가 실제 검색 결과에 있었나 (환각 여부)
  rank          그 청크의 검색 순위 (1 이 최상위). 모델이 상위를 고르는지 아무거나 고르는지
  shared        주장(mechanism)과 인용 청크가 공유하는 **도메인 용어**
  grounded      주장의 도메인 용어 중 인용 청크에도 있는 비율 — 근거가 실제로 뒷받침하는 정도

도메인 용어는 코퍼스에서 뽑는다: 코퍼스 전체에 흔한 말(문서빈도가 높은 것)과 너무 짧은 토큰을
빼고 남은 것. 불용어 목록을 손으로 적지 않으므로 코퍼스가 바뀌어도 따라간다.
"""
import argparse, collections, csv, json, math, pathlib, re, sys

from . import rag

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 도메인 용어로 인정할 조건
MIN_LEN = 3           # 이보다 짧은 토큰은 버린다 (조사·관사)
MAX_DF_RATIO = 0.30   # 코퍼스 청크의 이 비율보다 흔하면 일반어로 본다

_VOCAB = None

# ORCID·ISSN·DOI 조각 같은 식별자는 도메인 용어가 아니다. 숫자와 하이픈만으로 이뤄진 토큰을 뺀다.
_IDENT = re.compile(r"^[\d\-]+$|^\d[\d\-]{6,}")

def _is_identifier(w):
    return bool(_IDENT.match(w))



def domain_vocab(chunks=None):
    """코퍼스에서 도메인 용어 집합을 만든다 (문서빈도가 낮고 충분히 긴 토큰)."""
    global _VOCAB
    if _VOCAB is not None and chunks is None:
        return _VOCAB
    rows = chunks if chunks is not None else rag.load()
    if not rows:
        return set()
    df = collections.Counter()
    for c in rows:
        df.update(set(rag._tok(c["text"])))
    n = len(rows)
    vocab = {w for w, k in df.items()
             if len(w) >= MIN_LEN and k / n <= MAX_DF_RATIO and not _is_identifier(w)}
    if chunks is None:
        _VOCAB = vocab
    return vocab


def terms(text, vocab=None):
    """자유 서술에서 도메인 용어만 뽑는다."""
    v = vocab if vocab is not None else domain_vocab()
    return {w for w in rag._tok(str(text or "")) if w in v}


def _hypotheses(run):
    if run.get("mode") == "multi":
        h = (run.get("cause") or {}).get("hypotheses")
    else:
        h = ((run.get("single") or {}).get("q2") or {}).get("hypotheses")
    return [x for x in (h or []) if isinstance(x, dict)]


def _cited(run, h):
    """가설 하나의 인용을 원래 chunk_id 로. 익명화된 런(chunk_alias)은 D3 → 실제 id 로 되돌린다.
    대응표에 없는 id 는 그대로 둔다 — 환각 인용으로 잡혀야 하기 때문이다."""
    alias = run.get("chunk_alias") or {}
    return [alias.get(c, c) if isinstance(c, str) else c for c in (h.get("citations") or [])]


def citation_rows(run, vocab=None):
    """런 하나에서 인용 한 건마다 한 행. 인용이 없으면 빈 목록."""
    chunks = run.get("rag_chunks") or []
    by_id = {c["chunk_id"]: c for c in chunks}
    rank = {c["chunk_id"]: i + 1 for i, c in enumerate(chunks)}   # 검색 순위 (1 이 최상위)
    v = vocab if vocab is not None else domain_vocab()
    out = []
    for i, h in enumerate(_hypotheses(run)):
        claim = " ".join(str(h.get(k, "")) for k in ("issue", "mechanism", "predicted_trend"))
        ct = terms(claim, v)
        for cid in _cited(run, h):
            chunk = by_id.get(cid)
            shared = ct & terms(chunk["text"], v) if chunk else set()
            # 기준선: 검색된 6개 중 아무거나 인용했다면 근거율이 얼마였을까.
            # 실제 근거율이 이 값과 비슷하면, 인용이 내용을 보고 고른 것이 아니라는 뜻이다.
            base = [len(ct & terms(c["text"], v)) / len(ct) for c in chunks] if ct else []
            out.append(dict(
                run_id=run.get("run_id"), series=run.get("series"), model=run.get("model"),
                mode=run.get("mode"), rag=run.get("rag"), hypothesis=i, chunk_id=cid,
                in_retrieved=chunk is not None,
                rank=rank.get(cid),
                claim_terms=len(ct),
                shared=len(shared),
                grounded=round(len(shared) / len(ct), 3) if ct else None,
                grounded_random=round(sum(base) / len(base), 3) if base else None,
                grounded_best=round(max(base), 3) if base else None,
                shared_terms=" ".join(sorted(shared)[:12]),
            ))
    return out


def uncited_ranks(run):
    """인용되지 **않은** 청크의 순위. 선택도를 볼 때 비교 기준이 된다."""
    chunks = run.get("rag_chunks") or []
    cited = {c for h in _hypotheses(run) for c in _cited(run, h)}
    return [i + 1 for i, c in enumerate(chunks) if c["chunk_id"] not in cited]


def collect(paths):
    rows, runs = [], []
    v = domain_vocab()
    for p in paths:
        for f in sorted(pathlib.Path(p).glob("*.json")):
            run = json.loads(f.read_text(encoding="utf-8"))
            if not run.get("rag"):
                continue                      # rag off 런은 인용할 청크가 없다
            run["_dir"] = str(p)
            runs.append(run)
            for r in citation_rows(run, v):
                r["dir"] = str(p)
                r["file"] = f.name
                rows.append(r)
    return rows, runs


def summarize(rows, runs, by=("dir",)):
    key = lambda d: tuple(d.get(b) for b in by)
    agg = collections.defaultdict(lambda: dict(runs=0, cited_runs=0, cites=0, halluc=0,
                                               rank_sum=0, rank_n=0, grounded=[],
                                               g_rand=[], g_best=[], unc_rank=[]))
    for run in runs:
        a = agg[key(dict(run, dir=run["_dir"]))]
        a["runs"] += 1
        if any(h.get("citations") for h in _hypotheses(run)):
            a["cited_runs"] += 1
        a["unc_rank"] += uncited_ranks(run)
    for r in rows:
        a = agg[key(r)]
        a["cites"] += 1
        a["halluc"] += not r["in_retrieved"]
        if r["rank"]:
            a["rank_sum"] += r["rank"]; a["rank_n"] += 1
        if r["grounded"] is not None:
            a["grounded"].append(r["grounded"])
        if r.get("grounded_random") is not None:
            a["g_rand"].append(r["grounded_random"]); a["g_best"].append(r["grounded_best"])
    out = {}
    for k, a in agg.items():
        out[k] = dict(
            runs=a["runs"], cited_runs=a["cited_runs"], cites=a["cites"], halluc=a["halluc"],
            cited_rank=(a["rank_sum"] / a["rank_n"]) if a["rank_n"] else None,
            uncited_rank=(sum(a["unc_rank"]) / len(a["unc_rank"])) if a["unc_rank"] else None,
            grounded=(sum(a["grounded"]) / len(a["grounded"])) if a["grounded"] else None,
            grounded_random=(sum(a["g_rand"]) / len(a["g_rand"])) if a["g_rand"] else None,
            grounded_best=(sum(a["g_best"]) / len(a["g_best"])) if a["g_best"] else None,
        )
        # 선택도: 0 이면 아무거나 고른 것, 1 이면 검색된 것 중 가장 잘 맞는 것을 고른 것.
        v = out[k]
        if v["grounded"] is not None and v["grounded_random"] is not None \
                and v["grounded_best"] is not None:
            span = v["grounded_best"] - v["grounded_random"]
            v["selectivity"] = round((v["grounded"] - v["grounded_random"]) / span, 3) if span else None
        else:
            v["selectivity"] = None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="인용 적합성을 코드로만 잰다 (LLM 없음)")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--by", nargs="+", default=["dir"],
                    help="요약 기준 (dir/model/mode/series)")
    ap.add_argument("--csv", help="인용 한 건마다 한 행으로 저장")
    a = ap.parse_args(argv)

    rows, runs = collect(a.dirs)
    if not runs:
        print("rag on 런이 없습니다", file=sys.stderr); return 1
    if a.csv:
        cols = ["dir", "file", "series", "model", "mode", "hypothesis", "chunk_id",
                "in_retrieved", "rank", "claim_terms", "shared", "grounded",
                "grounded_random", "grounded_best", "shared_terms"]
        with open(a.csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"인용 {len(rows)}건 → {a.csv}")

    head = " / ".join(a.by)
    print(f"\n{head:34} {'런':>4} {'인용런':>6} {'인용':>5} {'환각':>5} "
          f"{'인용순위':>8} {'미인용순위':>10} {'근거율':>7} {'무작위':>7} {'최선':>7} {'선택도':>7}")
    for k, v in sorted(summarize(rows, runs, tuple(a.by)).items(),
                       key=lambda x: [str(y) for y in x[0]]):
        cr = f"{v['cited_rank']:.2f}" if v["cited_rank"] else "   -"
        ur = f"{v['uncited_rank']:.2f}" if v["uncited_rank"] else "   -"
        g = f"{v['grounded']:.3f}" if v["grounded"] is not None else "   -"
        gr = f"{v['grounded_random']:.3f}" if v["grounded_random"] is not None else "   -"
        gb = f"{v['grounded_best']:.3f}" if v["grounded_best"] is not None else "   -"
        sel = f"{v['selectivity']:.2f}" if v.get("selectivity") is not None else "   -"
        print(f"{' / '.join(str(x) for x in k):34} {v['runs']:>4} {v['cited_runs']:>6} "
              f"{v['cites']:>5} {v['halluc']:>5} {cr:>8} {ur:>10} {g:>7} {gr:>7} {gb:>7} {sel:>7}")
    print("\n인용순위·미인용순위: 1 이 검색 최상위. 인용순위가 더 낮으면(작으면) 모델이 상위 청크를 골랐다는 뜻.")
    print("근거율: 주장의 도메인 용어 중 인용 청크에도 있는 비율.")
    print("무작위: 검색된 6개 중 아무거나 인용했을 때의 기대값. **근거율이 이 값과 비슷하면 내용을 보고 고른 것이 아니다.**")
    print("최선: 검색된 6개 중 가장 잘 맞는 것을 골랐다면 나왔을 값 (천장).")
    print("선택도: (근거율-무작위)/(최선-무작위). 0 이면 아무거나 고른 것, 1 이면 최선을 고른 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
