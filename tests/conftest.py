import json, pathlib, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

@pytest.fixture
def series():
    def _load(sid):
        return json.loads((ROOT / "data" / "series" / f"{sid}.json").read_text(encoding="utf-8"))
    return _load

@pytest.fixture
def rules_of(series):
    """시리즈 하나에 대한 결함별 코드 규칙 판정."""
    from harc.checker import check
    from harc.defects.registry import ALL
    def _rules(sid):
        s = series(sid)
        chk = check(s)
        return {d.name: d.rule(s, chk["judgeable"][d.name]) for d in ALL}
    return _rules

@pytest.fixture(autouse=True)
def _quiet_log(tmp_path, monkeypatch):
    """테스트가 저장소의 logs/calls.jsonl 을 건드리지 않게 한다."""
    from harc import llm
    monkeypatch.setattr(llm, "LOG", tmp_path / "calls.jsonl")
