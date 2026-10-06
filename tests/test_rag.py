"""RAG: 평가 대상 논문 누수 필터와 corpus 안전장치."""
import json
import pytest
from harc import rag


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """chunks.jsonl 과 corpus/pdfs 를 임시로 깔아 둔다."""
    chunks = tmp_path / "chunks.jsonl"
    rows = [
        {"chunk_id": "lee2026#0", "paper_id": "lee2026",
         "text": "chiller temperature bowing sidewall passivation C3H2F6 taper profile"},
        {"chunk_id": "other2021#0", "paper_id": "other2021",
         "text": "bowing sidewall passivation fluorocarbon polymer deposition taper"},
    ]
    chunks.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(rag, "CHUNKS", chunks)
    monkeypatch.setattr(rag, "_INDEX", None)
    pdfs = tmp_path / "pdfs"; pdfs.mkdir()
    monkeypatch.setattr(rag, "PDF_DIR", pdfs)
    series = tmp_path / "series"; series.mkdir()
    monkeypatch.setattr(rag, "SERIES_DIR", series)
    monkeypatch.setattr(rag, "LIST_PATH", tmp_path / "문헌목록.csv")
    write_list(tmp_path, ["lee2026", "coatings2023", "kwon2024nano"])
    return tmp_path


def write_list(root, keys, bom=False):
    rows = "\n".join(f"{k},DOE시리즈,10.0/{k},다운로드" for k in keys)
    text = "key,분류,DOI_또는_ID,상태\n" + rows + "\n"
    (root / "문헌목록.csv").write_text(text, encoding="utf-8-sig" if bom else "utf-8")


def write_pdfs(root, keys, reindex=True):
    """PDF 를 깔고, 기본적으로 색인도 같은 논문 집합으로 맞춘다 (= build 를 돌린 상태).

    reindex=False 면 PDF 만 두어 '색인이 PDF 보다 오래된' 상태를 만든다.
    """
    for k in keys:
        (root / "pdfs" / f"{k}.pdf").write_bytes(b"%PDF-1.4")
    if reindex:
        write_chunks(root, keys)


def write_chunks(root, keys):
    """색인을 주어진 논문 집합으로 다시 쓴다."""
    rows = [{"chunk_id": f"{k}#0", "paper_id": k,
             "text": "bowing sidewall passivation taper profile"} for k in keys]
    (root / "chunks.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    rag._INDEX = None


def write_series(root, **sid_to_source):
    for sid, src in sid_to_source.items():
        (root / "series" / f"{sid}.json").write_text(
            json.dumps({"series_id": sid, "source_set": src}), encoding="utf-8")


def unlisted_warnings(ws):
    return [w for w in ws if "문헌목록에 없는" in w]


def missing_source_warnings(ws):
    return [w for w in ws if "source_set 인데" in w]


def unindexed_warnings(ws):
    return [w for w in ws if "색인에 없는 논문" in w]


def orphan_index_warnings(ws):
    return [w for w in ws if "PDF 가 없는 논문" in w]


def test_excluded_paper_never_retrieved(corpus):
    hits = rag.retrieve("bowing sidewall passivation taper", k=5, exclude_papers=["lee2026"])
    assert hits, "제외 필터가 전부를 걸러 버리면 안 된다"
    assert {h["paper_id"] for h in hits} == {"other2021"}


def test_without_exclusion_the_paper_is_retrievable(corpus):
    ids = {h["paper_id"] for h in rag.retrieve("bowing sidewall passivation taper", k=5)}
    assert "lee2026" in ids   # 필터가 없을 때는 나온다 → 위 테스트가 필터 덕분임을 보장


def test_source_sets_read_from_series():
    assert {"coatings2023", "lee2026"} <= rag.source_sets()


def test_listed_pdf_names_are_not_warned(corpus):
    """목록에 key 가 있으면 시리즈 source_set 이 아니어도 (배경 논문) 경고하지 않는다."""
    write_pdfs(corpus, ["lee2026", "coatings2023", "kwon2024nano"])
    write_series(corpus, S1="coatings2023", S2="lee2026")
    assert rag.corpus_warnings() == []


def test_unlisted_pdf_is_warned(corpus):
    write_pdfs(corpus, ["lee2026", "Lee_et_al_2026_final"])
    write_series(corpus, S2="lee2026")
    ws = unlisted_warnings(rag.corpus_warnings())
    assert len(ws) == 1
    assert "Lee_et_al_2026_final.pdf" in ws[0]
    assert "lee2026.pdf" not in ws[0]      # 목록에 있는 파일은 경고 대상이 아니다


def test_missing_source_set_pdf_is_warned(corpus):
    """S2 의 근거 논문을 아직 못 받은 상태 — 조용히 넘어가면 안 된다."""
    write_pdfs(corpus, ["coatings2023"])
    write_series(corpus, S1="coatings2023", S2="lee2026")
    ws = missing_source_warnings(rag.corpus_warnings())
    assert len(ws) == 1
    assert "lee2026" in ws[0]
    assert "coatings2023" not in ws[0]


def test_present_source_set_is_not_warned(corpus):
    write_pdfs(corpus, ["coatings2023", "lee2026"])
    write_series(corpus, S1="coatings2023", S2="lee2026")
    assert missing_source_warnings(rag.corpus_warnings()) == []


def test_two_warning_kinds_are_independent(corpus):
    """① 목록에 없는 파일과 ② 파일 없는 source_set 은 동시에 잡힌다."""
    write_pdfs(corpus, ["scan_01"])
    write_series(corpus, S2="lee2026")
    ws = rag.corpus_warnings()
    assert len(unlisted_warnings(ws)) == 1 and "scan_01.pdf" in unlisted_warnings(ws)[0]
    assert len(missing_source_warnings(ws)) == 1 and "lee2026" in missing_source_warnings(ws)[0]


def test_missing_list_file_is_warned_and_files_are_not(corpus):
    """목록이 없으면 그 사실만 알리고, 파일을 싸잡아 경고하지는 않는다."""
    (corpus / "문헌목록.csv").unlink()
    write_pdfs(corpus, ["lee2026", "scan_01"])
    ws = rag.corpus_warnings()
    assert any("문헌목록 없음" in w for w in ws)
    assert unlisted_warnings(ws) == []


def test_list_keys_tolerates_bom_and_blanks(corpus):
    """Excel 로 다시 저장해 BOM 이 붙어도 key 를 읽어야 한다."""
    write_list(corpus, ["lee2026", "", "  coatings2023  "], bom=True)
    assert rag.list_keys() == {"lee2026", "coatings2023"}


def test_repo_corpus_matches_its_literature_list():
    """저장소 실제 상태: corpus/pdfs 의 모든 파일이 문헌목록에 key 로 있어야 한다."""
    keys = rag.list_keys()
    assert keys and "coatings2023" in keys
    assert unlisted_warnings(rag.corpus_warnings()) == []


def test_missing_index_is_warned(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "CHUNKS", tmp_path / "없음.jsonl")
    monkeypatch.setattr(rag, "PDF_DIR", tmp_path / "pdfs")
    ws = rag.corpus_warnings()
    assert any("색인 없음" in w for w in ws)
    assert any("pdfs" in w for w in ws)


def test_new_pdf_without_rebuild_is_warned(corpus):
    """오늘 있었던 상황: PDF 를 새로 넣고 build 를 안 돌리면 그 논문은 검색에 없다."""
    write_pdfs(corpus, ["coatings2023"])                       # 색인 = {coatings2023}
    write_pdfs(corpus, ["coatings2023", "lee2026"], reindex=False)   # lee2026 만 색인 밖
    write_series(corpus, S1="coatings2023", S2="lee2026")
    ws = unindexed_warnings(rag.corpus_warnings())
    assert len(ws) == 1
    assert "lee2026" in ws[0] and "build" in ws[0]
    assert "coatings2023" not in ws[0]      # 이미 색인된 논문은 경고 대상이 아니다


def test_unindexed_paper_is_really_absent_from_search(corpus):
    """경고가 가리키는 손해를 검색으로 확인 — 색인 밖 논문은 retrieve 에 안 나온다."""
    write_pdfs(corpus, ["coatings2023"])
    write_pdfs(corpus, ["coatings2023", "lee2026"], reindex=False)
    assert "lee2026" not in {h["paper_id"] for h in rag.retrieve("bowing taper", k=5)}


def test_index_in_sync_is_not_warned(corpus):
    """build 를 돌린 뒤 — 색인 관련 경고가 없어야 한다."""
    write_pdfs(corpus, ["coatings2023", "lee2026"])
    write_series(corpus, S1="coatings2023", S2="lee2026")
    ws = rag.corpus_warnings()
    assert unindexed_warnings(ws) == [] and orphan_index_warnings(ws) == []


def test_deleted_pdf_leaves_orphan_chunks_and_is_warned(corpus):
    """PDF 를 지워도 청크가 남으면 사라진 논문이 계속 검색된다."""
    write_pdfs(corpus, ["coatings2023", "lee2026"])
    (corpus / "pdfs" / "lee2026.pdf").unlink()
    write_series(corpus, S1="coatings2023")
    ws = orphan_index_warnings(rag.corpus_warnings())
    assert len(ws) == 1 and "lee2026" in ws[0]


def test_missing_index_does_not_repeat_as_unindexed(corpus):
    """색인 파일이 아예 없으면 '색인 없음' 만 알리고 논문마다 다시 경고하지 않는다."""
    write_pdfs(corpus, ["coatings2023", "lee2026"], reindex=False)
    (corpus / "chunks.jsonl").unlink()
    ws = rag.corpus_warnings()
    assert any("색인 없음" in w for w in ws)
    assert unindexed_warnings(ws) == []


def test_indexed_papers_skips_broken_lines(corpus):
    """깨진 줄이 섞여도 나머지 줄로 대조를 계속한다."""
    (corpus / "chunks.jsonl").write_text(
        '{"chunk_id": "a#0", "paper_id": "a", "text": "x"}\n'
        '{"paper_id": \n'                      # 잘린 JSON
        '\n'                                   # 빈 줄
        '{"chunk_id": "b#0", "text": "paper_id 키 없음"}\n'
        '{"chunk_id": "c#0", "paper_id": "c", "text": "y"}\n', encoding="utf-8")
    assert rag.indexed_papers() == {"a", "c"}


def test_repo_index_is_in_sync_with_its_pdfs():
    """저장소 실제 상태: corpus/pdfs 와 corpus/chunks.jsonl 이 같은 논문 집합이어야 한다."""
    ws = rag.corpus_warnings()
    assert unindexed_warnings(ws) == [] and orphan_index_warnings(ws) == []


# --- 청크 크기 상한 (BM25 가 점수 매긴 본문 = 모델이 보는 본문) ---

def test_build_never_exceeds_the_word_cap(tmp_path, monkeypatch):
    """문단 하나가 상한보다 길어도 청크가 상한을 넘으면 안 된다.

    예전 판은 문단을 쌓다가 넘으면 끊기만 해서, 긴 문단 하나가 그대로 청크가 됐다.
    그래서 청크 중앙값이 584단어까지 커졌고 pipeline 이 1,200자로 잘라
    색인 본문의 30%만 모델에 도달했다.
    """
    long_para = " ".join(f"w{i}" for i in range(1000))      # 상한의 3배 짜리 문단 하나
    fake_pdf = tmp_path / "p1.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    class FakePage:
        def get_text(self): return long_para + "\n\n" + " ".join(f"x{i}" for i in range(50))
    monkeypatch.setattr(rag, "CHUNKS", tmp_path / "chunks.jsonl")
    import pymupdf
    monkeypatch.setattr(pymupdf, "open", lambda *a, **k: [FakePage()])

    rag.build(pdf_dir=tmp_path, max_words=100)
    chunks = [json.loads(l) for l in (tmp_path / "chunks.jsonl").open(encoding="utf-8")]
    assert chunks
    assert all(len(c["text"].split()) <= 100 for c in chunks), \
        [len(c["text"].split()) for c in chunks]
    assert all(c["paper_id"] == "p1" for c in chunks)


def test_build_keeps_every_word(tmp_path, monkeypatch):
    """쪼개되 버리지 않는다 — 본문이 통째로 색인에 들어가야 한다."""
    words = [f"w{i}" for i in range(500)]

    class FakePage:
        def get_text(self): return " ".join(words)
    monkeypatch.setattr(rag, "CHUNKS", tmp_path / "chunks.jsonl")
    import pymupdf
    monkeypatch.setattr(pymupdf, "open", lambda *a, **k: [FakePage()])
    (tmp_path / "p1.pdf").write_bytes(b"%PDF-1.4")

    rag.build(pdf_dir=tmp_path, max_words=120)
    got = " ".join(json.loads(l)["text"] for l in (tmp_path / "chunks.jsonl").open(encoding="utf-8"))
    assert got.split() == words


def test_repo_index_respects_the_cap():
    """저장소 실제 색인: 어떤 청크도 전달 상한을 넘지 않아야 한다.
    넘으면 BM25 가 점수 매긴 본문과 모델이 보는 본문이 달라진다."""
    from harc.pipeline import CHUNK_CHAR_CAP
    for c in rag.load():
        assert len(c["text"]) <= CHUNK_CHAR_CAP, c["chunk_id"]
        assert len(c["text"].split()) <= rag.CHUNK_WORDS, c["chunk_id"]
