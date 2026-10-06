"""전문가 평가 도구 — 눈가림이 새지 않는지, multi·single 이 같은 보고서 모양으로 나오는지."""
import csv
import io
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _runs():
    out = sorted((ROOT / "outputs_s5s8").glob("S6_*multi_rag-on*.json"))[:1] \
        + sorted((ROOT / "outputs_s5s8").glob("S8_*single_rag-on*.json"))[:1]
    assert len(out) == 2
    return out


def test_reports_are_blind_and_complete(tmp_path):
    runs = _runs()
    subprocess.run([sys.executable, "tools/expert_report.py", *map(str, runs), "--out", str(tmp_path),
                    "--seed", "3"], cwd=str(ROOT), check=True, capture_output=True)
    pages = sorted(tmp_path.glob("R*.html"))
    assert len(pages) == 2
    for p in pages:
        t = p.read_text(encoding="utf-8")
        for leak in ("gemini", "gemma", "google", "flash", "provider", "outputs_", ".json"):
            assert leak not in t.lower(), f"{p.name} 에 눈가림 정보가 샜다: {leak}"
        for section in ("1. 시리즈", "2. 조건별 수치", "3. 판단", "4. 원인 후보", "5. 다음 스플릿 제안"):
            assert section in t, (p.name, section)
        assert "$" not in t.split("<body>")[1].replace("data:image", ""), "LaTeX 조각이 남았다"
    key = list(csv.DictReader(io.open(tmp_path / "_key.csv", encoding="utf-8-sig")))
    assert {k["file"] for k in key} == {r.name for r in runs}
    sheet = list(csv.DictReader(io.open(tmp_path / "평가지.csv", encoding="utf-8-sig")))
    assert [s["report"] for s in sheet] == ["R01", "R02"] and "q_window" in sheet[0]
    assert "model" not in sheet[0] and "file" not in sheet[0]


def test_rubric_sheet_covers_every_series():
    out = subprocess.run([sys.executable, "tools/rubric_sheet.py"], cwd=str(ROOT), check=True,
                         capture_output=True, text=True).stdout
    for p in (ROOT / "answers").glob("S*.json"):
        a = json.loads(p.read_text(encoding="utf-8"))
        assert f"## {p.stem} — {a['source']['paper_id']}" in out
    assert "I20 (ICP · O2 20 sccm)" in out          # 목록 라벨은 첫 표현으로 보인다
