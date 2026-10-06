# 문헌 코퍼스 — 공개본에는 없음

RAG 실험에 쓴 문헌 PDF 와 검색 색인(`chunks.jsonl`, `embeddings.jsonl`, `placebo_chunks.jsonl`)은 저작권 때문에 싣지 않았다.
일부는 구독 저널이고, 일부는 CC BY-NC-ND 다.

재구성:
1. `literature/문헌목록.csv` 의 DOI·OA 링크로 PDF 를 받아 `corpus/pdfs/<key>.pdf` 로 저장한다 (key 는 CSV 의 첫 열)
2. `python -m harc.rag build` — BM25 색인
3. 임베딩·하이브리드 검색을 쓰려면 `python -m harc.rag build-embed`, 위약 실험은 `python -m harc.rag build-placebo`

저장된 런 JSON 의 `rag_chunks` 에는 청크 id 만 남아 있다. 채점(`harc.judge`)과 인용 적합성(`harc.cite_eval`)의 id 기반
부분은 코퍼스 없이 재현된다.
