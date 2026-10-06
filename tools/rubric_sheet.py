"""루브릭 검토지 — answers/*.json 을 전문가가 읽고 표시할 수 있는 Markdown 한 장으로 편다.

루브릭 8편은 전부 Claude 가 저자 결론을 읽고 쓴 **초안**이다(`rubric_author`). 정답·허용·오답의
경계에 판단이 섞여 있어, 전문가 평가의 첫 과제로 이 문서를 검토받는다. 검토 결과는 루브릭 파일에
반영하고 `rubric_author` 에 검토자와 날짜를 남긴다.

씀:  python tools/rubric_sheet.py > expert/루브릭검토.md
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIELDS = [("primary_issue", "가장 중대한 문제"), ("onset_condition", "시작 조건"), ("process_window", "공정 창")]


def label(cid, conds):
    v = conds.get(cid)
    v = v[0] if isinstance(v, list) else v
    return f"{cid} ({v})" if v is not None and str(v) != cid else str(cid)


def ids(xs, conds):
    return ", ".join(label(x, conds) for x in xs) if xs else "—"


def series_block(sid):
    a = json.loads((ROOT / "answers" / f"{sid}.json").read_text(encoding="utf-8"))
    s = json.loads((ROOT / "data" / "series" / f"{sid}.json").read_text(encoding="utf-8"))
    src, conds = a.get("source", {}), a.get("conditions") or {}
    out = [f"## {sid} — {src.get('paper_id', '')}", "",
           f"- **구조**: {s.get('structure')}", f"- **바꾼 변수**: {s.get('swept_variable')}",
           f"- **목표 (모델에 주는 입력 — 결론이 새지 않는지도 봐 주세요)**: {s.get('goal')}",
           f"- **DOI**: {src.get('doi', '—')}", "",
           "**저자 결론 (요약)**", "", f"> {a.get('authors_conclusion', '')}", ""]
    if src.get("rubric_author"):
        out += ["**초안 작성자가 확인을 요청한 판단**", "", f"> {src['rubric_author']}", ""]
    out += ["| 항목 | 정답 | 허용 (절반) | 오답 |", "|---|---|---|---|"]
    for key, name in FIELDS:
        r = (a.get("q1") or {}).get(key) or {}
        isc = key != "primary_issue"
        f = (lambda xs: ids(xs, conds)) if isc else (lambda xs: ", ".join(xs) if xs else "—")
        out.append(f"| {name} | {f(r.get('correct'))} | {f(r.get('acceptable'))} | {f(r.get('incorrect'))} |")
    out.append("")
    for key, name in FIELDS:
        note = ((a.get("q1") or {}).get(key) or {}).get("note")
        if note:
            out.append(f"- **{name} 근거**: {note}")
    out += ["", "**검토** (해당하는 곳에 표시)", "",
            "- [ ] 그대로 동의",
            "- [ ] 고칠 것 — 항목·조건과 이유:", "", "&nbsp;", "", "---", ""]
    return "\n".join(out)


def main():
    sids = sorted(p.stem for p in (ROOT / "answers").glob("S*.json"))
    head = ["# 루브릭 검토지", "",
            "각 시리즈(논문 1편)에 대해 **저자 결론을 기준으로 정답·허용·오답을 나눈 초안**입니다. 모델의 답은 이 표로 채점됩니다.",
            "", "- **정답** — 저자가 고른 것", "- **허용 (절반)** — 저자와 다르지만 데이터로 보아 타당한 것",
            "- **오답** — 데이터나 저자 결론과 어긋나는 것. 하나라도 들어가면 오답",
            "- 표에 없는 답은 채점하지 않습니다", "",
            "가장 확인이 필요한 것은 **공정 창과 시작 조건의 허용·오답 경계**입니다. 시간이 부족하면 각 시리즈의",
            "'초안 작성자가 확인을 요청한 판단' 만 보셔도 됩니다.", "", "---", ""]
    sys.stdout.write("\n".join(head) + "\n".join(series_block(s) for s in sids))


if __name__ == "__main__":
    main()
