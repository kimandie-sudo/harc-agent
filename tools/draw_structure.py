"""에이전트 구조도 — docs/구조도.svg 를 만든다.

씀:  python tools/draw_structure.py docs/구조도.svg
PNG 는 브라우저로 렌더링한다 (예: Playwright 스크린숏, 1660×750, 배율 2).
구조가 바뀌면(결함 모듈 수, 단계, 호출 수) 이 파일의 문구도 함께 고친다.
"""
import sys
W, H = 1660, 750
FONT = "'Pretendard','Noto Sans KR','Apple SD Gothic Neo','Malgun Gothic','WenQuanYi Zen Hei',sans-serif"
INK, INK2, MUTED, LINE = "#1f2328", "#4a5260", "#8a919c", "#3d4450"
GREY_BG, GREY = "#f1f2f4", "#6b7380"
ROLE = {  # 에이전트 역할 색 (채움, 테두리)
    "defect": ("#fdebd9", "#e09a5a"),
    "agg":    ("#e6e8fc", "#7d88dc"),
    "cause":  ("#fbe4ec", "#de8aa6"),
    "next":   ("#e3f3e3", "#6fb07a"),
}
o = []
def t(x, y, s, size=12, w=400, fill=INK, anchor="start", italic=False):
    st = ' font-style="italic"' if italic else ""
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}"{st}>{s}</text>')

def robot(x, y, c):
    o.append(f'<line x1="{x+7}" y1="{y-4}" x2="{x+7}" y2="{y}" stroke="{c}" stroke-width="1.4"/>'
             f'<circle cx="{x+7}" cy="{y-5}" r="1.6" fill="{c}"/>'
             f'<rect x="{x}" y="{y}" width="14" height="11" rx="3" fill="#fff" stroke="{c}" stroke-width="1.4"/>'
             f'<circle cx="{x+4.5}" cy="{y+5.5}" r="1.4" fill="{c}"/><circle cx="{x+9.5}" cy="{y+5.5}" r="1.4" fill="{c}"/>')

def gear(x, y, c=GREY):
    cx, cy = x + 7, y + 7
    import math
    for k in range(8):
        a = k * math.pi / 4
        o.append(f'<line x1="{cx+5*math.cos(a):.1f}" y1="{cy+5*math.sin(a):.1f}" x2="{cx+7.5*math.cos(a):.1f}" '
                 f'y2="{cy+7.5*math.sin(a):.1f}" stroke="{c}" stroke-width="2.2" stroke-linecap="round"/>')
    o.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="#fff" stroke="{c}" stroke-width="1.6"/>'
             f'<circle cx="{cx}" cy="{cy}" r="1.8" fill="{c}"/>')

def book(x, y, c=GREY):
    o.append(f'<path d="M{x},{y+2} q6,-3 8,0 v12 q-2,-3 -8,0 z M{x+16},{y+2} q-6,-3 -8,0 v12 q2,-3 8,0 z" '
             f'fill="#fff" stroke="{c}" stroke-width="1.4" stroke-linejoin="round"/>')

def node(x, y, w, label, kind, sub=None, icon=None):
    """kind: 'rule' 또는 ROLE 키. y 는 중심선."""
    h = 32
    bg, st = (GREY_BG, GREY) if kind == "rule" else ROLE[kind]
    o.append(f'<rect x="{x}" y="{y-h/2}" width="{w}" height="{h}" rx="6" fill="{bg}" stroke="{st}" stroke-width="1.3"/>')
    ix = x + 10
    if kind == "rule":
        (book if icon == "book" else gear)(ix, y - 7)
    else:
        robot(ix, y - 5, st)
    t(ix + 22, y + 4.5, label, 12.5, 600)
    if sub:
        for k, s in enumerate(sub if isinstance(sub, list) else [sub]):
            t(x + w / 2, y + h / 2 + 16 + 14 * k, s, 10.5, 400, MUTED, "middle", italic=True)

def doc(x, y, lines):
    """x,y = 아이콘 중심."""
    x0, y0 = x - 9, y - 12
    o.append(f'<path d="M{x0},{y0} h12 l6,6 v18 h-18 z M{x0+12},{y0} v6 h6" fill="#fff" stroke="{GREY}" stroke-width="1.2" stroke-linejoin="round"/>')
    for k in range(3):
        o.append(f'<line x1="{x0+4}" y1="{y0+11+4*k}" x2="{x0+14}" y2="{y0+11+4*k}" stroke="{GREY}" stroke-width="1"/>')
    for k, s in enumerate(lines):
        t(x, y + 26 + 13 * k, s, 10.5, 400, INK2, "middle")

def img_icon(x, y):
    o.append(f'<rect x="{x}" y="{y}" width="26" height="20" rx="2" fill="#fff" stroke="{GREY}" stroke-width="1.3"/>'
             f'<polyline points="{x+3},{y+17} {x+10},{y+9} {x+15},{y+14} {x+19},{y+10} {x+23},{y+17}" fill="none" stroke="{GREY}" stroke-width="1.3"/>'
             f'<circle cx="{x+19}" cy="{y+6}" r="2" fill="{GREY}"/>')
def table_icon(x, y):
    o.append(f'<rect x="{x}" y="{y}" width="26" height="20" rx="2" fill="#fff" stroke="{GREY}" stroke-width="1.3"/>')
    for k in (1, 2):
        o.append(f'<line x1="{x}" y1="{y+k*20/3:.1f}" x2="{x+26}" y2="{y+k*20/3:.1f}" stroke="{GREY}" stroke-width="1"/>')
    o.append(f'<line x1="{x+9}" y1="{y}" x2="{x+9}" y2="{y+20}" stroke="{GREY}" stroke-width="1"/>')
def target_icon(x, y):
    for r in (10, 6, 2):
        o.append(f'<circle cx="{x+13}" cy="{y+10}" r="{r}" fill="{"#fff" if r > 2 else GREY}" stroke="{GREY}" stroke-width="1.3"/>')

def arrow(x1, y1, x2, y2):
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" stroke-width="1.3" marker-end="url(#ah)"/>')
def curve(x1, y1, x2, y2, head=True):
    mx = (x1 + x2) / 2
    m = ' marker-end="url(#ah)"' if head else ""
    o.append(f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" fill="none" stroke="{LINE}" stroke-width="1.3"{m}/>')

# ── 제목과 범례
t(28, 38, "HARC Agent — 에이전트 구조", 20, 700)
t(28, 60, "수치는 규칙이 계산하고, LLM 에이전트는 그 결과를 취합해 결정을 보조한다", 12.5, 400, INK2)
lx = W - 470
o.append(f'<rect x="{lx}" y="24" width="150" height="26" rx="6" fill="{ROLE["agg"][0]}" stroke="{ROLE["agg"][1]}"/>')
robot(lx + 9, 32, ROLE["agg"][1]); t(lx + 30, 41.5, "LLM 에이전트", 12, 600)
t(lx, 68, "모두 같은 모델 — 역할별 지시문만 다름", 10.5, 400, MUTED, italic=True)
o.append(f'<rect x="{lx+260}" y="24" width="150" height="26" rx="6" fill="{GREY_BG}" stroke="{GREY}"/>')
gear(lx + 269, 30); t(lx + 290, 41.5, "규칙 (코드)", 12, 600)
t(lx + 260, 68, "같은 입력이면 항상 같은 결과", 10.5, 400, MUTED, italic=True)

MID = 340
# ── 입력
ins = [(MID - 92, img_icon, "단면 SEM"), (MID, table_icon, "조건별 측정값"), (MID + 92, target_icon, "목표·설계 요구")]
for y, fn, lab in ins:
    fn(40, y - 18)
    t(53, y + 18, lab, 10.5, 400, INK2, "middle")
    curve(72, y - 8, 110, MID, head=False)
arrow(110, MID, 122, MID)
node(124, MID, 118, "데이터 점검", "rule", ["빠진 값·해상도 한계"])
arrow(242, MID, 268, MID)
doc(282, MID, ["판단 가능", "항목"])

# ── 결함 모듈 (펼침)
BX, BW = 360, 520
boxes = [(98, "보잉"), (258, "네킹"), (520, "낮은 종횡비")]
for top, name in boxes:
    bh = 128; ym = top + 64
    o.append(f'<rect x="{BX}" y="{top}" width="{BW}" height="{bh}" rx="4" fill="#ffffff" stroke="#d5d9df"/>')
    t(BX + 10, top + 18, f"결함 모듈 · {name}", 11, 700, INK2)
    node(BX + 18, ym, 112, "규칙 계산", "rule", ["변화율·임계값"])
    arrow(BX + 130, ym, BX + 158, ym)
    doc(BX + 174, ym, ["수치·방향", "최악/최선 조건"])
    arrow(BX + 196, ym, BX + 226, ym)
    node(BX + 228, ym, 128, "결함 해석", "defect", ["규칙 수치를 읽고 서술"])
    arrow(BX + 356, ym, BX + 392, ym)
    doc(BX + 410, ym, ["심각도·근거"])
    curve(318, MID, BX - 2, ym)                          # 펼침
    curve(BX + BW, ym, 918, MID, head=False)              # 모음
t(BX + BW / 2, 432, "⋮", 22, 700, MUTED, "middle")
t(BX + BW / 2, 456, "결함 모듈 10종 — 테이퍼·언더컷·측벽각·식각 정지·마스크 침식·CD 편차·깊이 손실 …", 11, 400, MUTED, "middle")
t(BX + BW / 2, 474, "서로 결과를 보지 않으므로 병렬로 돈다", 10.5, 400, MUTED, "middle", italic=True)

# ── 취합 → 원인 → 다음 스플릿
arrow(918, MID, 930, MID)
node(932, MID, 128, "취합 에이전트", "agg")
node(932, MID + 120, 128, "설계 요구·우열", "rule", ["요구 충족 판정·파레토"])
arrow(996, MID + 104, 996, MID + 18)
arrow(1060, MID, 1092, MID)
doc(1110, MID, ["중대 문제·", "시작 조건·공정 창"])
arrow(1132, MID, 1160, MID)
node(1162, MID, 128, "원인 에이전트", "cause")
node(1162, MID + 120, 128, "문헌 검색", "rule", ["24편 · 평가 논문 제외"], icon="book")
arrow(1226, MID + 104, 1226, MID + 18)
arrow(1290, MID, 1318, MID)
doc(1336, MID, ["원인 후보·", "문헌 인용"])
arrow(1358, MID, 1386, MID)
node(1388, MID, 172, "다음 스플릿 에이전트", "next")
arrow(1560, MID, 1584, MID)
doc(1604, MID, ["판단 보고서"])
t(1604, MID + 56, "→ 전문가 2명 평가", 10.5, 400, MUTED, "middle", italic=True)

# ── 한 줄 요약
t(W / 2, 704, "데이터 점검 → 결함 모듈 ×10 (규칙 계산 → 해석 에이전트) → 취합 에이전트 → 원인 에이전트 (+문헌) → 다음 스플릿 에이전트",
  15, 400, INK2, "middle")
t(W / 2, 730, "런마다 LLM 호출 13회 · 단계 사이는 JSON 으로만 주고받는다", 12, 400, MUTED, "middle")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">'
       f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
       f'<path d="M0,1 L9,5 L0,9" fill="none" stroke="{LINE}" stroke-width="1.6"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>' + "".join(o) + "</svg>")
open(sys.argv[1], "w", encoding="utf-8").write(svg)
