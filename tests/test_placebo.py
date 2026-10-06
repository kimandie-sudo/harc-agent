"""위약 청크 — 기제 실험의 세 번째 팔 (DESIGN_기제실험.md).

"검색된 문헌이 해로운가" 와 "프롬프트가 길어서 해로운가" 를 가르려면
**길이는 같고 내용은 무관한** 텍스트가 필요하다. 그 세 조건을 고정한다:
  ① RAG 코퍼스와 한 건도 겹치지 않는다
  ② 길이가 실제 검색 결과에 맞는다
  ③ 결정론적이다 (같은 런이면 같은 위약)
전부 API 키 없이 돈다.
"""
import json
import pathlib

import pytest

from harc import rag

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def pool(tmp_path, monkeypatch):
    """길이가 뚜렷이 다른 위약 후보들."""
    chunks = [
        {"chunk_id": "xa#0", "paper_id": "xa", "text": " ".join(["alpha"] * 10)},
        {"chunk_id": "xa#1", "paper_id": "xa", "text": " ".join(["beta"] * 50)},
        {"chunk_id": "xb#0", "paper_id": "xb", "text": " ".join(["gamma"] * 100)},
        {"chunk_id": "xb#1", "paper_id": "xb", "text": " ".join(["delta"] * 300)},
    ]
    p = tmp_path / "placebo.jsonl"
    p.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks),
                 encoding="utf-8")
    monkeypatch.setattr(rag, "PLACEBO_CHUNKS", p)
    monkeypatch.setattr(rag, "_PLACEBO", None)
    return chunks


def _hit(n, cid="real#0"):
    return {"chunk_id": cid, "paper_id": cid.split("#")[0], "text": " ".join(["w"] * n)}


def test_picks_the_nearest_length(pool):
    got = rag.placebo_for([_hit(52), _hit(11)])
    assert [c["chunk_id"] for c in got] == ["xa#1", "xa#0"]


def test_does_not_reuse_a_chunk_within_one_run(pool):
    """같은 길이를 두 번 요구해도 같은 청크를 두 번 주지 않는다."""
    got = rag.placebo_for([_hit(50), _hit(50), _hit(50)])
    ids = [c["chunk_id"] for c in got]
    assert len(ids) == len(set(ids)) == 3


def test_is_deterministic(pool):
    hits = [_hit(40), _hit(120), _hit(9)]
    assert [c["chunk_id"] for c in rag.placebo_for(hits)] == \
           [c["chunk_id"] for c in rag.placebo_for(hits)]


def test_returns_the_same_count_as_the_real_hits(pool):
    for n in (1, 2, 4):
        assert len(rag.placebo_for([_hit(50)] * n)) == n


def test_runs_out_gracefully(pool):
    """후보보다 많이 요구하면 있는 만큼만 준다 — 같은 청크를 재사용하지 않는다."""
    got = rag.placebo_for([_hit(50)] * 10)
    assert len(got) == len(pool)


def test_empty_pool_returns_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "PLACEBO_CHUNKS", tmp_path / "없음.jsonl")
    monkeypatch.setattr(rag, "_PLACEBO", None)
    assert rag.placebo_for([_hit(50)]) == []
    w = rag.placebo_warnings()
    assert w and "build-placebo" in w[0]


def test_marks_its_own_provenance(pool):
    for c in rag.placebo_for([_hit(50)]):
        assert c["retriever"] == "placebo" and c["score"] is None


# ── 실제 색인에 대한 검사 ────────────────────────────────────────────────
REAL = ROOT / "corpus" / "placebo_chunks.jsonl"


@pytest.mark.skipif(not REAL.exists(), reason="위약 색인 없음 — rag build-placebo 필요")
def test_real_placebo_never_overlaps_the_rag_corpus():
    """**이게 깨지면 실험이 무의미하다.** 위약이 검색 코퍼스에 있으면 대조가 안 된다."""
    assert rag.placebo_warnings() == []
    placebo = {c["paper_id"] for c in rag.placebo_load()}
    corpus = {c["paper_id"] for c in rag.load()}
    assert not (placebo & corpus), f"겹치는 논문: {placebo & corpus}"


@pytest.mark.skipif(not REAL.exists(), reason="위약 색인 없음")
def test_real_placebo_matches_length_closely():
    """실제 코퍼스 질의로 길이 매칭 품질을 잰다. 합계 단어 차가 5% 이내여야 한다."""
    hits = rag.retrieve("bowing sidewall passivation fluorocarbon", k=6, retriever="bm25")
    if not hits:
        pytest.skip("RAG 색인이 비어 있음")
    got = rag.placebo_for(hits)
    a = sum(len(rag._tok(h["text"])) for h in hits)
    b = sum(len(rag._tok(c["text"])) for c in got)
    assert abs(b - a) / a < 0.05, f"길이 차 {abs(b-a)/a:.1%} (실제 {a}, 위약 {b})"


@pytest.mark.skipif(not REAL.exists(), reason="위약 색인 없음")
def test_pipeline_records_the_chunk_source(tmp_path):
    """런 JSON 에 chunk_source 가 남아야 나중에 팔을 구분할 수 있다."""
    import json as _json
    from harc.pipeline import run
    series = _json.loads((ROOT / "data" / "series" / "S2.json").read_text(encoding="utf-8"))
    out = run(series, provider="mock", mode="multi", use_rag=True,
              base_dir=str(ROOT / "data"), chunk_source="placebo")
    assert out["chunk_source"] == "placebo"
    ids = {c["chunk_id"].split("#")[0] for c in out["rag_chunks"]}
    corpus = {c["paper_id"] for c in rag.load()}
    assert ids and not (ids & corpus), f"위약 런에 코퍼스 청크가 섞였다: {ids & corpus}"
