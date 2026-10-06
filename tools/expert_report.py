"""전문가 평가용 보고서 — 런 JSON 하나를 엔지니어가 읽을 한 장짜리 HTML 로 그린다.

PROJECT.md 의 주 지표는 "전문가 2명이 시스템을 직접 써 보고 출력 품질을 평가" 하는 것이다.
전문가가 보는 것은 런 JSON 이 아니라 이 보고서다. 그래서:

- **눈가림.** 모델·제공사·RAG 여부·파일명을 싣지 않는다. 보고서는 R01, R02 … 로만 부른다.
  대응표(`_key.csv`)는 따로 쓰고, 전문가에게 보내지 않는다
- **순서 무작위.** 같은 시리즈의 보고서가 붙어 나오지 않게 고정 시드로 섞는다
- **multi 와 single 을 같은 모양으로.** 둘의 답 위치가 달라도 보고서의 절은 같다
- **근거를 따라갈 수 있게.** 규칙이 계산한 수치 표와, 인용한 문헌 청크의 원문을 함께 싣는다
- 원 논문 제목·저자·결론은 싣지 않는다 — 시리즈 파일에 원래 없다

라이선스: S2(lee2026)·S5(you2025)의 SEM 은 CC BY-NC-ND 다. 보고서는 평가자에게만 보내고 공개하지 않는다.

씀:
    python tools/expert_report.py outputs_v3base/*multi_rag-on*.json --out expert/reports --seed 1
    python tools/expert_report.py <런 JSON> --out /tmp/r --no-shuffle      # 한 장 미리보기
"""
import argparse
import base64
import csv
import html
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harc.defects.registry import ALL  # noqa: E402

LABEL = {d.name: d.label for d in ALL}
METRIC = {d.name: d.metric for d in ALL}

# 평가지 항목. 1~5 점(5 가 가장 좋음). 문구를 바꾸면 이미 받은 응답과 비교할 수 없게 되니 주의.
ITEMS = [
    ("q_issue", "가장 중대한 문제를 맞게 짚었는가"),
    ("q_window", "추천한 공정 창(쓸 조건)이 타당한가"),
    ("q_cause", "원인 후보가 그럴듯하고, 데이터와 맞게 지지·배제했는가"),
    ("q_next", "다음 스플릿 제안을 실제로 쓸 만한가"),
    ("q_trace", "결론의 근거(수치·문헌)를 따라가 확인할 수 있는가"),
    ("q_use", "이 보고서를 실제 스플릿 검토에 쓰겠는가"),
]


def tex(s):
    """모델 출력의 LaTeX 조각을 기호로 바꾼다 ($\\to$, $\\mu$m …)."""
    if s is None:
        return ""
    s = str(s)
    for a, b in ((r"\to", "→"), (r"\rightarrow", "→"), (r"\mu", "µ"), (r"\uparrow", "↑"),
                 (r"\downarrow", "↓"), (r"\times", "×"), (r"\approx", "≈"), (r"\sim", "~"),
                 (r"\le", "≤"), (r"\ge", "≥"), (r"\%", "%"), (r"\circ", "°")):
        s = s.replace(a, b)
    return s.replace("$", "")


def esc(s):
    return html.escape(tex(s))


def view(run):
    """multi·single 의 답을 같은 모양으로 꺼낸다. 스키마를 벗어난 답도 죽지 않고 문자열로 보인다."""
    if run.get("mode") == "multi":
        agg = run.get("aggregate") or {}
        nxt = run.get("next") or {}
        rec = nxt.get("recommendation") if isinstance(nxt, dict) else nxt
        hyps = (run.get("cause") or {}).get("hypotheses") if isinstance(run.get("cause"), dict) else run.get("cause")
        return {
            "primary_issue": agg.get("primary_issue") if isinstance(agg, dict) else agg,
            "onset": agg.get("onset_condition") if isinstance(agg, dict) else None,
            "window": agg.get("process_window") if isinstance(agg, dict) else None,
            "tradeoffs": agg.get("tradeoffs") if isinstance(agg, dict) else None,
            "ranking": agg.get("ranking") if isinstance(agg, dict) else None,
            "hyps": hyps or [],
            "cannot": (run.get("cause") or {}).get("cannot_assess") if isinstance(run.get("cause"), dict) else None,
            "rec": rec if isinstance(rec, dict) else {"direction": rec},
            "alternatives": nxt.get("alternatives") if isinstance(nxt, dict) else None,
            "measure_next": nxt.get("measure_next") if isinstance(nxt, dict) else None,
        }
    s = run.get("single") or {}
    q1 = s.get("q1") if isinstance(s.get("q1"), dict) else {"primary_issue": s.get("q1")}
    q2 = s.get("q2") if isinstance(s.get("q2"), dict) else {"hypotheses": s.get("q2")}
    q3 = s.get("q3") if isinstance(s.get("q3"), dict) else {"direction": s.get("q3")}
    return {
        "primary_issue": q1.get("primary_issue"), "onset": q1.get("onset_condition"),
        "window": q1.get("process_window"), "tradeoffs": q1.get("tradeoffs"), "ranking": None,
        "hyps": q2.get("hypotheses") or [], "cannot": None, "rec": q3,
        "alternatives": None, "measure_next": None,
    }


def as_list(v):
    if v in (None, "", [], {}):
        return []
    return v if isinstance(v, list) else [v]


def fmt(v):
    if isinstance(v, bool):
        return "중단" if v else "정상"
    if isinstance(v, list):
        return ", ".join(tex(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {tex(x)}" for k, x in v.items())
    return tex(v)


def data_table(run, series):
    """조건 × 판정 가능한 결함 지표. 규칙(코드)이 계산한 값이며 모델 출력이 아니다."""
    rules = run.get("rules") or {}
    cols = [n for n, r in rules.items()
            if isinstance(r, dict) and any(isinstance(x, dict) and x.get(METRIC.get(n)) is not None
                                           for x in r.get("rows", []))]
    vals = {}
    for n in cols:
        for row in rules[n]["rows"]:
            if isinstance(row, dict):
                vals[(row.get("condition"), n)] = row.get(METRIC[n])
    head = "".join(f"<th>{esc(LABEL.get(n, n))}<br><small>{esc(METRIC[n])}</small></th>" for n in cols)
    body = []
    for c in series["conditions"]:
        cells = "".join(f"<td>{esc(fmt(vals.get((c['id'], n), '—')))}</td>" for n in cols)
        body.append(f"<tr><th>{esc(c['id'])}</th><td>{esc(c.get('level'))}</td>{cells}</tr>")
    missing = [LABEL.get(n, n) for n, r in rules.items() if n not in cols]
    note = (f"<p class=muted>판정 불가(값 없음): {esc(', '.join(missing))}</p>" if missing else "")
    return (f"<table><thead><tr><th>조건</th><th>수준</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>{note}")


def images(series):
    out = []
    for c in series["conditions"]:
        p = ROOT / "data" / (c.get("image") or "")
        if c.get("image") and p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            out.append(f"<figure><img src='data:image/png;base64,{b64}' alt='{esc(c['id'])}'>"
                       f"<figcaption>{esc(c['id'])}<br><small>{esc(c.get('level'))}</small></figcaption></figure>")
    return f"<div class=sem>{''.join(out)}</div>" if out else "<p class=muted>이 시리즈에는 단면 SEM 이 없다.</p>"


def chunk_lookup(run):
    """인용 id → 청크 원문. 익명화한 런(D1…)이면 대응표로 되돌린다."""
    chunks = {c["chunk_id"]: c.get("text", "") for c in (run.get("rag_chunks") or []) if isinstance(c, dict)}
    alias = run.get("chunk_alias") or {}
    return lambda cid: chunks.get(alias.get(cid, cid))


def constraint_note(run):
    """v5: 규칙이 판정한 설계 요구와, 모델이 그것을 어겼는지."""
    cons, chk = run.get("constraints"), run.get("constraint_check") or {}
    if not cons:
        return ""
    viol = chk.get("violations") or []
    head = (f"<p><b>설계 요구</b>: {esc(', '.join(cons['constraints']))} — 충족 조건 {esc(', '.join(cons['eligible']))}</p>")
    if viol:
        head += (f"<p style='color:#b00020'><b>⚠ 공정 창에 설계 요구를 못 맞추는 조건이 들어 있다</b>: {esc(', '.join(viol))}</p>")
    return head


def render(run, series, rid):
    v = view(run)
    look = chunk_lookup(run)
    cited = []
    hyp_rows = []
    for h in as_list(v["hyps"]):
        if not isinstance(h, dict):
            hyp_rows.append(f"<li>{esc(h)}</li>")
            continue
        refs = []
        for cid in as_list(h.get("citations")):
            if cid not in cited:
                cited.append(cid)
            refs.append(f"[{cited.index(cid) + 1}]")
        hyp_rows.append(
            "<li><b>" + esc(h.get("mechanism")) + "</b>"
            + (f" <span class=tag>{esc(h.get('issue'))}</span>" if h.get("issue") else "")
            + f"<br>예상: {esc(h.get('predicted_trend'))}<br>관찰: {esc(h.get('observed_trend'))}"
            + f"<br>판정: <b>{esc(h.get('verdict'))}</b> {' '.join(refs)}</li>")
    refs_html = "".join(
        f"<li><b>[{i + 1}]</b> <code>{esc(cid)}</code><blockquote>{esc((look(cid) or '(원문 없음 — 검색 결과에 없는 인용)')[:700])}</blockquote></li>"
        for i, cid in enumerate(cited))
    ranking = "".join(
        f"<tr><td>{esc(r.get('defect'))}</td><td>{esc(r.get('severity'))}</td><td>{esc(r.get('why'))}</td></tr>"
        for r in as_list(v["ranking"]) if isinstance(r, dict))
    rec = v["rec"] or {}
    alt = "".join(f"<li>{esc(fmt(a))}</li>" for a in as_list(v["alternatives"]))
    return f"""<!doctype html>
<html lang=ko><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>보고서 {rid}</title>
<style>
body{{font-family:-apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1d1d1f;background:#fff;line-height:1.55}}
h1{{font-size:1.5em;margin-bottom:0}} h2{{font-size:1.15em;border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:2em}}
table{{border-collapse:collapse;width:100%;font-size:.88em;margin:8px 0}} th,td{{border:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}}
thead th{{background:#f4f4f6}} small,.muted{{color:#666}} .sem{{display:flex;flex-wrap:wrap;gap:10px}}
figure{{margin:0;text-align:center;font-size:.8em}} figure img{{height:220px;border:1px solid #ccc}}
.box{{background:#f7f9fc;border:1px solid #dbe3ef;border-radius:6px;padding:10px 14px}}
.tag{{background:#eef;border-radius:4px;padding:0 6px;font-size:.85em}} blockquote{{margin:4px 0 10px 12px;color:#444;font-size:.88em;white-space:pre-wrap}}
@media print{{h2{{break-after:avoid}} figure img{{height:160px}}}}
</style></head><body>
<h1>보고서 {rid}</h1>
<p class=muted>공정 시리즈 분석 결과 — 규칙(코드)이 수치를 계산하고, 언어 모델이 그 결과를 취합했다.</p>

<h2>1. 시리즈</h2>
<table><tr><th>구조</th><td>{esc(series.get('structure'))}</td></tr>
<tr><th>목표</th><td>{esc(series.get('goal'))}</td></tr>
<tr><th>바꾼 변수</th><td>{esc(series.get('swept_variable'))}</td></tr>
<tr><th>고정 조건</th><td>{esc(series.get('other_fixed'))}</td></tr></table>
{images(series)}

<h2>2. 조건별 수치 (규칙 계산)</h2>
{data_table(run, series)}

<h2>3. 판단</h2>
<div class=box>
{constraint_note(run)}
<p><b>가장 중대한 문제</b>: {esc(fmt(v['primary_issue']))}</p>
<p><b>시작 조건</b>: {esc(fmt(v['onset'])) or '<span class=muted>(답 없음)</span>'}</p>
<p><b>공정 창 (쓸 조건)</b>: {esc(fmt(v['window'])) or '<span class=muted>(답 없음)</span>'}</p>
<p><b>트레이드오프</b></p><ul>{''.join(f'<li>{esc(fmt(t))}</li>' for t in as_list(v['tradeoffs']))}</ul>
</div>
{f'<table><thead><tr><th>결함</th><th>중대도(0–3)</th><th>이유</th></tr></thead><tbody>{ranking}</tbody></table>' if ranking else ''}

<h2>4. 원인 후보</h2>
<ol>{''.join(hyp_rows) or '<li class=muted>(없음)</li>'}</ol>
{f'<p class=muted>이 데이터로 판단할 수 없는 것: {esc(fmt(v["cannot"]))}</p>' if v['cannot'] else ''}
{f'<h3>인용한 문헌</h3><ol class=refs>{refs_html}</ol>' if cited else ''}

<h2>5. 다음 스플릿 제안</h2>
<div class=box>
<p><b>바꿀 변수</b>: {esc(fmt(rec.get('variable')))}</p>
<p><b>방향</b>: {esc(fmt(rec.get('direction')))}</p>
<p><b>제안 수준</b>: {esc(fmt(rec.get('suggested_levels')))}</p>
{f"<p><b>기대 효과</b>: {esc(fmt(rec.get('expected_effect')))}</p>" if rec.get('expected_effect') else ''}
<p><b>위험</b>: {esc(fmt(rec.get('risk')))}</p>
</div>
{f'<p><b>대안</b></p><ul>{alt}</ul>' if alt else ''}
{f'<p><b>다음에 꼭 잴 것</b>: {esc(fmt(v["measure_next"]))}</p>' if v['measure_next'] else ''}
</body></html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="전문가 평가용 눈가림 보고서를 만든다")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--prefix", default="R")
    a = ap.parse_args(argv)
    runs = sorted(a.runs)
    if not a.no_shuffle:
        random.Random(a.seed).shuffle(runs)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key, sheet = [], []
    for i, f in enumerate(runs, 1):
        run = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        series = json.loads((ROOT / "data" / "series" / f"{run['series']}.json").read_text(encoding="utf-8"))
        rid = f"{a.prefix}{i:02d}"
        (out / f"{rid}.html").write_text(render(run, series, rid), encoding="utf-8")
        key.append({"report": rid, "file": pathlib.Path(f).name, "series": run["series"],
                    "provider": run.get("provider"), "model": run.get("model"), "mode": run.get("mode"),
                    "rag": run.get("rag"), "pipeline_version": run.get("pipeline_version")})
        sheet.append({"report": rid, **{k: "" for k, _ in ITEMS}, "minutes": "", "wrong_statements": "",
                      "comment": ""})
    with open(out / "_key.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader(); w.writerows(key)
    with open(out / "평가지.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0]))
        w.writeheader(); w.writerows(sheet)
    print(f"{len(runs)}장 → {out}  (대응표 _key.csv 는 평가자에게 보내지 말 것)")


if __name__ == "__main__":
    main()
