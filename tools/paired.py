"""짝 비교 — 셀(시리즈×모델×모드[×검색기]) 안에서 rag off 대 on 을 맞대고 부호검정한다.

MANIFEST_rag.md 의 "on / 무 / off" 표를 내는 코드다. 그동안 배치마다 손으로 셌는데,
그러다 열을 잘못 읽어 README 인용표를 7/7 틀렸다고 보고한 적이 있다 (2026-09-22).

**대조군에는 검색기 축이 없다.** rag=off 런은 검색을 안 하므로 retriever 값이 무의미하다.
그래서 셀 키에서 검색기를 떼고 맞춘다 — 같은 off 런이 세 검색기 각각의 짝이 된다.
짝들이 완전히 독립은 아니라는 뜻이고, 부호검정 p 는 그만큼 낙관적으로 읽어야 한다.

**모드를 나눠 봐야 한다.** multi 는 aggregate(q1) 가 검색보다 먼저 돈다 — 청크는
cause(q2) 에만 들어가고 next(q3) 에는 cause 의 출력을 거쳐 간접으로 닿는다.
그래서 multi 의 q1 은 rag on/off 가 **같은 입력**이고, 거기서 갈리는 쌍은 서버 비결정성뿐이다.
multi q1 은 효과 측정이 아니라 **잡음 바닥**이다 (--mode 로 나눠 볼 것).

씀: python tools/paired.py outputs_retriever6
    python tools/paired.py outputs_retriever6 --mode single
"""
import collections
import csv
import io
import json
import math
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRIMARY_FIELDS = ("q1.primary_issue", "q1.process_window", "q1.onset_condition",
                  "q3.variable", "q3.direction")


def score_dir(d, tmp):
    out = pathlib.Path(tmp) / f"{pathlib.Path(d).name}.csv"
    subprocess.run([sys.executable, "-m", "harc.judge", d, "--csv", str(out)],
                   cwd=str(ROOT), check=True, capture_output=True)
    return list(csv.DictReader(io.open(out, encoding="utf-8-sig")))


def meta(d):
    """judge 의 CSV 에는 retriever/chunk_source 열이 없어 런 JSON 에서 직접 읽는다."""
    m = {}
    for f in (ROOT / d).glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        m[f.name] = (r.get("retriever", "bm25"), r.get("chunk_source", "rag"))
    return m


def sign_test(w, l):
    """양측 부호검정. 동률은 이미 뺀 상태로 받는다."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def cells_of(rows, mt, fields=None, mode=None):
    per = collections.defaultdict(list)
    for r in rows:
        if r["judge"] != "code" or not r["score"]:
            continue
        if fields and r["field"] not in fields:
            continue
        if mode and r["mode"] != mode:
            continue
        per[r["file"]].append(float(r["score"]))
    by_file = {r["file"]: r for r in rows}
    cells = collections.defaultdict(dict)
    for f, ss in per.items():
        row = by_file[f]
        on = row["rag"] == "True"
        retr = mt.get(f, ("bm25", "rag"))[0]
        key = (row["series"], row["model"], row["mode"]) + ((retr,) if on else ())
        cells[key]["on" if on else "off"] = sum(ss) / len(ss)
    return cells


def compare(cells, retr=None):
    offs = {k: v["off"] for k, v in cells.items() if "off" in v and len(k) == 3}
    pairs = []
    for k, v in cells.items():
        if "on" not in v or len(k) != 4:
            continue
        if retr and k[3] != retr:
            continue
        if k[:3] in offs:
            pairs.append((offs[k[:3]], v["on"]))
    w = sum(1 for o, n in pairs if n > o)
    l = sum(1 for o, n in pairs if n < o)
    d = (sum(n - o for o, n in pairs) / len(pairs)) if pairs else 0.0
    return len(pairs), w, len(pairs) - w - l, l, d, sign_test(w, l)


def main(dirs, mode=None):
    with tempfile.TemporaryDirectory() as tmp:
        for d in dirs:
            rows, mt = score_dir(d, tmp), meta(d)
            print(f"\n===== {d}" + (f"  [mode={mode}]" if mode else "") + " =====")
            print(f"{'검색기':<10s} {'쌍':>3s} {'on/무/off':>12s} {'평균차':>8s} {'부호검정 p':>10s}")
            for r in (None, "bm25", "embed", "hybrid"):
                n, w, t, l, dd, p = compare(cells_of(rows, mt, mode=mode), r)
                if n:
                    print(f"{(r or '전체'):<10s} {n:3d} {f'{w}/{t}/{l}':>12s} {dd:+8.3f} {p:10.3f}")
            print("\n항목별 (탐색적):")
            for fld in PRIMARY_FIELDS:
                n, w, t, l, dd, p = compare(cells_of(rows, mt, {fld}, mode))
                if n:
                    print(f"  {fld:<22s} 쌍{n:3d}  {w}/{t}/{l:<3}  {dd:+.3f}  p={p:.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = None
    if "--mode" in args:
        i = args.index("--mode")
        mode = args[i + 1]
        args = args[:i] + args[i + 2:]
    main(args or ["outputs_retriever4"], mode)
