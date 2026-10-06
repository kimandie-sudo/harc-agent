"""임베딩·하이브리드 검색. 전부 API 키 없이 돈다 (mock 임베딩).

mock 임베딩은 무작위가 아니라 **토큰 해시 가산**이라 어휘가 겹치면 유사해진다.
그래서 '임베딩 검색이 관련 청크를 고른다' 를 실제로 검사할 수 있다.
"""
import json
import pytest

from harc import rag


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """세 논문짜리 작은 코퍼스. 주제가 뚜렷이 갈리게 만든다."""
    chunks = [
        {"chunk_id": "a#0", "paper_id": "a",
         "text": "sidewall bowing passivation fluorocarbon deposition protects the mask"},
        {"chunk_id": "a#1", "paper_id": "a",
         "text": "ion bombardment angle broadens the profile near the top of the hole"},
        {"chunk_id": "b#0", "paper_id": "b",
         "text": "etch stop occurs when polymer accumulates at the bottom of a deep contact"},
        {"chunk_id": "b#1", "paper_id": "b",
         "text": "selectivity to the amorphous carbon layer falls as temperature rises"},
        {"chunk_id": "leak#0", "paper_id": "leak",
         "text": "sidewall bowing passivation fluorocarbon deposition protects the mask"},
    ]
    cp = tmp_path / "chunks.jsonl"
    cp.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks),
                  encoding="utf-8")
    ep = tmp_path / "embeddings.jsonl"
    mp = tmp_path / "embeddings.meta.json"
    monkeypatch.setattr(rag, "CHUNKS", cp)
    monkeypatch.setattr(rag, "EMBEDS", ep)
    monkeypatch.setattr(rag, "EMBEDS_META", mp)
    monkeypatch.setattr(rag, "_EMB", None)
    monkeypatch.setattr(rag, "_INDEX", None)
    rag.embed_build(provider="mock")
    return chunks


def _ids(hits):
    return [h["chunk_id"] for h in hits]


@pytest.mark.parametrize("retriever", ["bm25", "embed", "hybrid"])
def test_finds_the_on_topic_chunk(corpus, retriever):
    hits = rag.retrieve("bowing passivation fluorocarbon", k=2,
                        retriever=retriever, embed_provider="mock")
    assert hits, f"{retriever} 가 아무것도 찾지 못했다"
    assert "a#0" in _ids(hits), f"{retriever} 가 주제가 맞는 청크를 놓쳤다: {_ids(hits)}"


@pytest.mark.parametrize("retriever", ["bm25", "embed", "hybrid"])
def test_leak_filter_holds_for_every_retriever(corpus, retriever):
    """누수 필터는 **순위를 매기기 전에** 걸려야 한다. 검색기마다 따로 새면 안 된다."""
    hits = rag.retrieve("bowing passivation fluorocarbon", k=5,
                        exclude_papers=["leak"], retriever=retriever, embed_provider="mock")
    assert "leak" not in {h["paper_id"] for h in hits}, \
        f"{retriever} 에서 제외 논문이 새어 나왔다: {_ids(hits)}"
    # leak#0 은 a#0 과 본문이 같으므로, 필터가 없으면 반드시 상위에 올라온다
    unfiltered = rag.retrieve("bowing passivation fluorocarbon", k=5,
                              retriever=retriever, embed_provider="mock")
    assert "leak#0" in _ids(unfiltered), "테스트 전제가 깨졌다 — 필터 없이도 안 나온다"


@pytest.mark.parametrize("retriever", ["bm25", "embed", "hybrid"])
def test_respects_k(corpus, retriever):
    assert len(rag.retrieve("etch", k=2, retriever=retriever, embed_provider="mock")) <= 2


def test_hybrid_fuses_both_rankings(corpus):
    """RRF 는 두 검색기가 함께 올린 것을 위로 올린다."""
    fused = rag._rrf([(9.0, 0), (8.0, 1)], [(0.9, 1), (0.8, 2)])
    order = [i for _, i in fused]
    assert order[0] == 1, f"양쪽에 든 후보가 1위여야 한다: {order}"
    assert set(order) == {0, 1, 2}


def test_retriever_is_recorded_on_hits(corpus):
    for rt in ("bm25", "embed", "hybrid"):
        h = rag.retrieve("etch stop polymer", k=1, retriever=rt, embed_provider="mock")
        assert h and h[0]["retriever"] == rt


def test_unknown_retriever_is_rejected(corpus):
    with pytest.raises(ValueError):
        rag.retrieve("etch", retriever="지어낸이름", embed_provider="mock")


def test_embed_index_staleness_is_caught(corpus, tmp_path):
    """청크를 바꾸고 build-embed 를 안 돌리면 경고가 나와야 한다.

    이게 없으면 임베딩이 옛 청크를 가리킨 채 조용히 엉뚱한 것을 검색한다.
    """
    assert rag.embed_warnings() == []
    rag.CHUNKS.write_text(
        rag.CHUNKS.read_text(encoding="utf-8")
        + json.dumps({"chunk_id": "c#0", "paper_id": "c", "text": "새 청크"},
                     ensure_ascii=False) + "\n", encoding="utf-8")
    w = rag.embed_warnings()
    assert w and "build-embed" in w[0], f"어긋남을 못 잡았다: {w}"


def test_missing_embed_index_is_caught(corpus):
    rag.EMBEDS.unlink()
    rag._EMB = None
    w = rag.embed_warnings()
    assert w and "임베딩 색인 없음" in w[0]


def test_embed_retrieval_without_index_returns_nothing(corpus):
    """색인이 없으면 조용히 빈 결과 — 파이프라인이 '청크 0개' 경고로 잡는다."""
    rag.EMBEDS.unlink()
    rag._EMB = None
    assert rag.retrieve("bowing", retriever="embed", embed_provider="mock") == []


def test_mock_embedding_is_deterministic_and_unit_length(corpus):
    a = rag._mock_embed("bowing passivation")
    b = rag._mock_embed("bowing passivation")
    assert a == b
    assert abs(sum(x * x for x in a) ** 0.5 - 1.0) < 1e-9


def test_similar_text_is_closer_than_unrelated(corpus):
    """mock 임베딩이 의미 있는 유사도를 준다는 전제를 지킨다."""
    q = rag._mock_embed("bowing passivation fluorocarbon")
    near = rag._mock_embed("bowing passivation fluorocarbon deposition")
    far = rag._mock_embed("selectivity amorphous carbon temperature")
    dot = lambda x, y: sum(a * b for a, b in zip(x, y))
    assert dot(q, near) > dot(q, far)


def test_build_resumes_instead_of_redoing_everything(corpus, monkeypatch):
    """분당 한도 때문에 빌드가 끊길 수 있다. 다시 돌리면 남은 것만 만들어야 한다."""
    calls = []
    real = rag._mock_embed
    monkeypatch.setattr(rag, "_mock_embed",
                        lambda t, dim=rag.EMBED_DIM: (calls.append(t), real(t, dim))[1])
    rag.embed_build(provider="mock")           # 이미 corpus fixture 가 한 번 만들었다
    assert calls == [], f"이미 있는 것을 다시 만들었다: {len(calls)}건"

    # 한 건을 지우면 그 한 건만 다시 만든다
    lines = rag.EMBEDS.read_text(encoding="utf-8").splitlines()
    rag.EMBEDS.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    rag._EMB = None
    rag.embed_build(provider="mock")
    assert len(calls) == 1, f"남은 1건만 만들어야 하는데 {len(calls)}건"
    assert rag.embed_warnings() == []


def test_build_without_resume_redoes_all(corpus, monkeypatch):
    calls = []
    real = rag._mock_embed
    monkeypatch.setattr(rag, "_mock_embed",
                        lambda t, dim=rag.EMBED_DIM: (calls.append(t), real(t, dim))[1])
    rag.embed_build(provider="mock", resume=False)
    assert len(calls) == len(corpus)
