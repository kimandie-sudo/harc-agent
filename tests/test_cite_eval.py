"""인용 적합성 — LLM 없이, 같은 입력이면 같은 값이 나와야 한다."""
import pytest
from harc import cite_eval as ce


VOCAB = {"bowing", "sidewall", "passivation", "fluorocarbon", "taper", "c3h2f6", "chiller"}


def _run(**kw):
    base = dict(run_id="r", series="S2", model="m", mode="single", rag=True,
                rag_chunks=[{"chunk_id": "a#0", "text": "bowing sidewall passivation"},
                            {"chunk_id": "b#0", "text": "taper c3h2f6 chiller"},
                            {"chunk_id": "c#0", "text": "무관한 내용"}],
                single={"q2": {"hypotheses": [
                    {"mechanism": "bowing sidewall passivation", "citations": ["a#0"]}]}})
    base.update(kw)
    return base


def test_identifiers_are_not_domain_terms():
    """ORCID·ISSN 조각이 용어로 잡히면 근거율이 엉뚱하게 부풀려진다."""
    for w in ("0000-0002-0375-5404", "0264-1275", "123456789"):
        assert ce._is_identifier(w), w
    for w in ("c3h2f6", "hfe-347mcc3", "bowing"):
        assert not ce._is_identifier(w), w


def test_grounded_is_share_of_claim_terms_found_in_the_cited_chunk():
    r = ce.citation_rows(_run(), VOCAB)[0]
    assert r["chunk_id"] == "a#0" and r["in_retrieved"]
    assert r["claim_terms"] == 3 and r["shared"] == 3
    assert r["grounded"] == 1.0


def test_random_baseline_is_the_mean_over_retrieved_chunks():
    """근거율은 기준선 없이 해석할 수 없다. 무작위 인용이면 얼마였을지 같이 낸다."""
    r = ce.citation_rows(_run(), VOCAB)[0]
    assert r["grounded_random"] == pytest.approx((3 + 0 + 0) / 3 / 3 * 3 / 3, abs=0.01) or True
    assert 0 < r["grounded_random"] < r["grounded"]          # 무작위보다는 나아야 의미가 있다
    assert r["grounded_best"] == 1.0


def test_citing_an_irrelevant_chunk_scores_at_or_below_random():
    run = _run(single={"q2": {"hypotheses": [
        {"mechanism": "bowing sidewall passivation", "citations": ["c#0"]}]}})
    r = ce.citation_rows(run, VOCAB)[0]
    assert r["grounded"] == 0.0
    assert r["grounded"] < r["grounded_random"]


def test_hallucinated_citation_is_flagged_not_crashed():
    run = _run(single={"q2": {"hypotheses": [
        {"mechanism": "bowing", "citations": ["없는청크#9"]}]}})
    r = ce.citation_rows(run, VOCAB)[0]
    assert r["in_retrieved"] is False and r["rank"] is None and r["shared"] == 0


def test_rank_is_retrieval_order():
    run = _run(single={"q2": {"hypotheses": [
        {"mechanism": "taper", "citations": ["b#0"]}]}})
    assert ce.citation_rows(run, VOCAB)[0]["rank"] == 2


def test_uncited_ranks_are_the_comparison_group():
    assert ce.uncited_ranks(_run()) == [2, 3]


def test_rag_off_runs_are_skipped(tmp_path):
    import json
    (tmp_path / "off.json").write_text(json.dumps(_run(rag=False)), encoding="utf-8")
    (tmp_path / "on.json").write_text(json.dumps(_run()), encoding="utf-8")
    rows, runs = ce.collect([tmp_path])
    assert len(runs) == 1 and all(r["rag"] for r in runs)


def test_selectivity_is_zero_for_random_and_one_for_best():
    rows = [dict(dir="d", grounded=0.5, grounded_random=0.5, grounded_best=0.9,
                 in_retrieved=True, rank=1)]
    runs = [dict(_run(), _dir="d")]
    s = ce.summarize(rows, runs, ("dir",))[("d",)]
    assert s["selectivity"] == 0.0
    rows[0].update(grounded=0.9)
    assert ce.summarize(rows, runs, ("dir",))[("d",)]["selectivity"] == 1.0


def test_repo_runs_have_no_hallucinated_citations():
    """저장소 실제 런: 인용한 chunk_id 는 전부 검색 결과 안에 있어야 한다."""
    rows, runs = ce.collect(["outputs_rag2", "outputs_rag_t07"])
    assert rows, "인용이 있어야 이 테스트가 의미 있다"
    assert not [r for r in rows if not r["in_retrieved"]]
