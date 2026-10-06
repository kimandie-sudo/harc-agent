"""문헌 RAG. 현재: BM25 키워드 검색 + 논문 단위 누수 필터. 다음 단계: 임베딩 검색과 하이브리드.

corpus/pdfs/<paper_id>.pdf  →  python -m harc.rag build  →  corpus/chunks.jsonl
paper_id 는 literature/문헌목록.csv 의 key — 누수 필터가 파일명으로 동작한다.
"""
import csv, hashlib, json, math, random, re, sys, time, pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "corpus" / "chunks.jsonl"
PDF_DIR = ROOT / "corpus" / "pdfs"
SERIES_DIR = ROOT / "data" / "series"
LIST_PATH = ROOT / "literature" / "문헌목록.csv"

def _tok(t):
    t = t.lower()
    # 화학식·숫자 토큰 보존 (c4f8, hfe-347mcc3, 20c 등)
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[가-힣]+", t)

def source_sets(series_dir=None):
    """data/series/*.json 이 참조하는 source_set(=논문 id) 모음."""
    out = set()
    for p in sorted(pathlib.Path(series_dir or SERIES_DIR).glob("*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if s.get("source_set"):
            out.add(s["source_set"])
    return out

def list_keys(list_path=None):
    """literature/문헌목록.csv 의 key 모음. 코퍼스에 들어와도 되는 논문 id 의 기준."""
    p = pathlib.Path(list_path or LIST_PATH)
    if not p.exists():
        return None                       # 목록 자체가 없음 — 호출자가 구분해 다룬다
    with p.open(encoding="utf-8-sig", newline="") as f:      # Excel 재저장 BOM 허용
        return {(r.get("key") or "").strip() for r in csv.DictReader(f)} - {""}

def indexed_papers(chunks_path=None):
    """색인에 실제로 들어 있는 paper_id 모음. 색인 파일이 없으면 None (= 대조 불가)."""
    p = pathlib.Path(chunks_path or CHUNKS)
    if not p.exists():
        return None
    out = set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.add(json.loads(line)["paper_id"])
            except (ValueError, KeyError):
                continue                  # 깨진 줄은 건너뛴다 — 대조는 남은 줄로 계속
    return out

def corpus_warnings(pdf_dir=None, series_dir=None, chunks_path=None, list_path=None,
                    check_embeddings=False):
    """RAG 안전장치: 색인 유무와 코퍼스-문헌목록 불일치를 미리 잡는다.

    corpus/pdfs/<paper_id>.pdf 의 paper_id 는 literature/문헌목록.csv 의 key 와 같아야 한다.
    어긋나면 retrieve(exclude_papers=[source_set]) 가 평가 대상 논문을 못 걸러 정답이 새어 들어간다.
    경고는 네 방향을 본다. ①② 는 corpus/pdfs ↔ 문헌목록, ③④ 는 corpus/pdfs ↔ 색인:
      ① 목록에 없는 파일        — 출처 불명이고 누수 필터가 잡지 못한다
      ② source_set 인데 파일 없음 — 해당 시리즈가 근거 논문 없이 돌아간다
      ③ PDF 는 있는데 색인에 없음 — PDF 를 새로 넣고 build 를 안 돌린 상태. 그 논문은 검색에 없다
      ④ 색인에 있는데 PDF 가 없음 — 지워진 논문의 청크가 검색에 남아 있다
    check_embeddings=True 면 임베딩 색인이 chunks.jsonl 을 따라오는지도 본다
    (--retriever embed/hybrid 를 쓸 때만 의미가 있어 기본은 끔).
    """
    pdf_dir = pathlib.Path(pdf_dir or PDF_DIR)
    chunks_path = pathlib.Path(chunks_path or CHUNKS)
    w = []
    if not chunks_path.exists():
        w.append(f"corpus 색인 없음: {chunks_path} — `python -m harc.rag build` 를 먼저 실행해야 RAG 가 동작합니다")
    if not pdf_dir.exists():
        w.append(f"corpus/pdfs 디렉터리 없음: {pdf_dir}")
        return w

    have = {p.stem for p in pdf_dir.glob("*.pdf")}

    known = list_keys(list_path)
    if known is None:
        w.append(f"문헌목록 없음: {pathlib.Path(list_path or LIST_PATH)} — "
                 "목록에 없는 PDF 를 가려낼 수 없습니다")
    else:
        unlisted = sorted(p.name for p in pdf_dir.glob("*.pdf") if p.stem not in known)
        if unlisted:
            w.append("corpus/pdfs 에 문헌목록에 없는 파일: " + ", ".join(unlisted)
                     + " — 파일명을 literature/문헌목록.csv 의 key 와 맞춰야"
                       " 누수 필터가 논문 단위로 동작합니다")

    missing = sorted(source_sets(series_dir) - have)
    if missing:
        w.append("시리즈 source_set 인데 corpus/pdfs 에 PDF 가 없음: " + ", ".join(missing)
                 + " — 해당 시리즈는 자기 논문을 근거로 쓰지 못한 채 실행됩니다")

    # 색인이 corpus/pdfs 를 따라오지 못한 경우. 색인 자체가 없을 때는 위에서 이미 알렸다.
    indexed = indexed_papers(chunks_path)
    if indexed is not None:
        unindexed = sorted(have - indexed)
        if unindexed:
            w.append("corpus/pdfs 에 있는데 색인에 없는 논문: " + ", ".join(unindexed)
                     + " — 색인이 PDF 보다 오래됐습니다."
                       " 이 논문들은 --rag on 이어도 검색되지 않으므로"
                       " `python -m harc.rag build` 를 다시 실행하세요")
        orphan = sorted(indexed - have)
        if orphan:
            w.append("색인에 있는데 corpus/pdfs 에 PDF 가 없는 논문: " + ", ".join(orphan)
                     + " — 지워진 논문의 청크가 검색에 남아 있습니다."
                       " `python -m harc.rag build` 를 다시 실행하세요")
    if check_embeddings:
        w += embed_warnings(chunks_path)
    return w

CHUNK_WORDS = 350          # 청크 하나의 단어 수 상한. pipeline._rag 의 전달 상한과 짝을 이룬다.

def _split_words(words, cap):
    """단어 목록을 cap 이하 조각으로 자른다."""
    return [words[i:i + cap] for i in range(0, len(words), cap)] or [[]]

def build(pdf_dir=None, max_words=None, out_path=None):
    """PDF → 청크. **max_words 를 넘는 청크를 만들지 않는다.**

    예전 판은 문단을 쌓다가 목표를 넘으면 끊었다. 문단 하나가 이미 목표보다 길면 그대로 통과해서
    청크 중앙값이 584단어(3,711자)까지 커졌고, pipeline._rag 가 1,200자로 자르는 바람에
    **색인 본문의 30%만 모델에 도달했다** (203청크 중 193개가 상한 초과).
    BM25 는 청크 전체로 점수를 매기므로, 뒷부분 용어로 상위에 오른 청크가 정작 그 부분이
    잘린 채 전달되는 일이 생긴다. 이제 긴 문단은 단어 단위로 쪼개 상한을 지킨다.
    """
    import pymupdf
    max_words = max_words or CHUNK_WORDS
    out = []
    for pdf in sorted(pathlib.Path(pdf_dir or PDF_DIR).glob("*.pdf")):
        pid = pdf.stem
        doc = pymupdf.open(pdf)
        text = "\n".join(p.get_text() for p in doc)
        text = re.split(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text)[0]  # 참고문헌 제외
        paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
        buf, n = [], 0

        def flush():
            nonlocal buf, n
            if buf:
                out.append({"chunk_id": f"{pid}#{n}", "paper_id": pid, "text": " ".join(buf)})
                n += 1
                buf = []

        for p in paras:
            if len(p) < 40:
                continue
            w = p.split()
            if len(w) > max_words:                 # 문단 자체가 상한보다 길다 → 쪼갠다
                flush()
                for piece in _split_words(w, max_words):
                    buf = [" ".join(piece)]
                    flush()
                continue
            if sum(len(x.split()) for x in buf) + len(w) > max_words:
                flush()
            buf.append(p)
        flush()
    dest = pathlib.Path(out_path or CHUNKS)
    dest.parent.mkdir(exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(out)} chunks from {len({c['paper_id'] for c in out})} papers → {dest}")
    if out_path is None:          # 위약 색인은 코퍼스 정합성 검사 대상이 아니다
        for msg in corpus_warnings(pdf_dir):
            print("WARN:", msg, file=sys.stderr)

# ── 위약 청크 ───────────────────────────────────────────────────────────────
# 기제 실험용 (DESIGN_기제실험.md). "검색된 문헌이 해로운가" 와 "프롬프트가 길어서 해로운가"
# 를 가르려면 **길이는 같고 내용은 무관한** 텍스트가 필요하다.
#   - literature/xai/ 의 XAI 논문을 쓴다. 실제 과학 산문이고, 이 도메인과 무관하고,
#     corpus/chunks.jsonl 에 들어간 적이 없다.
#   - 런마다 실제 검색이 돌려줬을 청크의 단어 수에 맞춰 고른다. 토큰 수가 거의 같아야
#     "길이 때문" 이라는 설명을 검정할 수 있다.

PLACEBO_PDF_DIR = ROOT / "literature" / "xai"
PLACEBO_CHUNKS = ROOT / "corpus" / "placebo_chunks.jsonl"

def placebo_build(pdf_dir=None, out_path=None):
    """literature/xai/*.pdf → corpus/placebo_chunks.jsonl."""
    build(pdf_dir or PLACEBO_PDF_DIR, out_path=out_path or PLACEBO_CHUNKS)

_PLACEBO = None
def placebo_load(path=None):
    global _PLACEBO
    p = pathlib.Path(path or PLACEBO_CHUNKS)
    if not p.exists():
        return []
    if _PLACEBO is None or _PLACEBO[1] != p.stat().st_mtime_ns:
        _PLACEBO = ([json.loads(l) for l in p.open(encoding="utf-8")], p.stat().st_mtime_ns)
    return _PLACEBO[0]

def placebo_for(hits, path=None):
    """실제 검색 결과 hits 의 단어 수에 맞춘 위약 청크를 같은 개수만큼 돌려준다.

    **결정론적이다** — 같은 hits 면 항상 같은 위약이 나온다. 한 런 안에서 같은 청크를
    두 번 쓰지 않는다. 길이가 가까운 것부터 집되, 동률이면 chunk_id 순으로 고른다.
    """
    pool = placebo_load(path)
    if not pool or not hits:
        return []
    wc = [(len(_tok(c["text"])), c) for c in pool]
    used, out = set(), []
    for h in hits:
        want = len(_tok(h["text"]))
        best = min((c for n, c in wc if c["chunk_id"] not in used),
                   key=lambda c: (abs(len(_tok(c["text"])) - want), c["chunk_id"]),
                   default=None)
        if best is None:
            break
        used.add(best["chunk_id"])
        out.append(dict(best, score=None, retriever="placebo"))
    return out

def placebo_warnings(path=None):
    """위약 색인이 있는지, 그리고 **RAG 코퍼스와 겹치지 않는지.**"""
    pool = placebo_load(path)
    if not pool:
        return ["위약 색인 없음: corpus/placebo_chunks.jsonl — "
                "`python -m harc.rag build-placebo` 를 실행해야 --chunk-source placebo 가 동작합니다"]
    overlap = {c["paper_id"] for c in pool} & {c["paper_id"] for c in load()}
    if overlap:
        return ["위약 색인이 RAG 코퍼스와 겹칩니다: " + ", ".join(sorted(overlap))
                + " — 위약은 검색 코퍼스에 없는 문헌이어야 합니다"]
    return []

# ── 심은 청크 (양성 대조군) ─────────────────────────────────────────────────
# 기제 실험 v2 (DESIGN_기제실험.md). "청크가 답을 안 바꿨다" 를 해석하려면 청크가 답을
# 바꿀 **수 있다**는 것부터 보여야 한다. 이 시리즈의 틀린 조건 하나를 공정 창이라고 주장하는
# 문장을 실제 검색 결과의 최하위 청크 **앞에 붙인다.** 청크를 통째로 바꾸지 않는 이유는
# 길이다 — 개수·위치·길이가 rag 팔과 같아야 '길이' 와 '내용' 이 섞이지 않는다.
#
# 표적은 data/planted.json 에 둔다. **채점용 정답 디렉터리는 읽지 않는다** — 정답 파일은
# 어떤 경로로도 프롬프트에 들어가면 안 된다. 라벨은 시리즈 파일(data/series/)에서 가져온다.

PLANTED_TARGETS = ROOT / "data" / "planted.json"
PLANTED_TEMPLATE = ("In a follow-up study of this process series, condition {target}{label} "
                    "gave the most vertical profile and was adopted as the process window.")
PLANTED_MARK = "planted:"          # rag_chunks 기록용 표지. 프롬프트에는 익명화된 id 만 들어간다

def planted_spec(series, path=None):
    """이 시리즈에 심을 표적과 문장. 표적이 없거나 시리즈에 없는 조건이면 ValueError."""
    p = pathlib.Path(path) if path else PLANTED_TARGETS
    targets = json.loads(p.read_text(encoding="utf-8"))
    sid = series["series_id"]
    target = targets.get(sid)
    if not target:
        raise ValueError(f"{p.name} 에 {sid} 의 표적이 없다")
    cond = {c["id"]: c for c in series.get("conditions", [])}
    if target not in cond:
        raise ValueError(f"표적 {target} 이 {sid} 의 조건이 아니다")
    lab = cond[target].get("label") or cond[target].get("level")
    sentence = PLANTED_TEMPLATE.format(target=target, label=f" ({lab})" if lab else "")
    return {"target": target, "label": lab, "sentence": sentence}

def planted_for(hits, series, path=None):
    """검색 결과의 **최하위** 청크 본문 앞에 심은 문장을 붙인다. 나머지는 그대로 둔다.

    그 청크의 chunk_id 앞에는 PLANTED_MARK 를 붙여 런 기록에서 알아볼 수 있게 한다 —
    그래서 **익명화 없이 쓰면 안 된다** (pipeline 이 막는다).
    """
    if not hits:
        return []
    spec = planted_spec(series, path)
    out = [dict(h) for h in hits]
    last = out[-1]
    last["text"] = spec["sentence"] + " " + last["text"]
    last["chunk_id"] = PLANTED_MARK + last["chunk_id"]
    last["retriever"] = "planted"
    return out

# ── 임베딩 검색 ─────────────────────────────────────────────────────────────
# BM25 는 어휘가 겹쳐야 찾는다. 리뷰 코퍼스 실험(outputs_reviews)에서 드러난 실패가
# 정확히 그 한계였다 — 길고 용어가 빽빽한 리뷰가 검색 창을 차지했으나 주장과 어휘가
# 겹치지 않아 근거율이 실험논문의 1/3 이었다. 임베딩은 그 가설을 직접 검정하는 수단이다.

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
EMBEDS = ROOT / "corpus" / "embeddings.jsonl"
EMBEDS_META = ROOT / "corpus" / "embeddings.meta.json"
RRF_K = 60                 # Reciprocal Rank Fusion 상수. 표준값.
FUSE_POOL = 50             # 각 검색기에서 몇 위까지 가져와 섞을지

def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else list(v)

def _mock_embed(text, dim=EMBED_DIM):
    """키 없이 도는 결정론적 임베딩. 토큰을 해시해 자리에 더한다.

    무작위가 아니라 **어휘가 겹치면 유사해지는** 벡터다. 그래야 테스트가
    '임베딩 검색이 관련 청크를 고른다'를 실제로 검사할 수 있다.
    """
    v = [0.0] * dim
    for w in _tok(text):
        h = int(hashlib.sha1(w.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
        v[(h >> 16) % dim] += 0.5      # 자리 하나만 쓰면 충돌이 너무 잦다
    return _unit(v)

# 무료 티어 임베딩 한도는 **분당 토큰**이다 (실측: 350단어 청크 50개 = 약 23k토큰 한 번이면 소진).
# 요청 수가 아니라 토큰이 기준이므로, 배치를 줄이는 것만으로는 해결되지 않고 분 단위로 쉬어야 한다.
EMBED_BATCH = 25           # 요청당 청크 수
EMBED_COOLDOWN = 62        # 429 를 받으면 분 한도가 리셋될 때까지 기다린다

def embed_texts(texts, task_type, provider="google", model=None, dim=EMBED_DIM,
                batch=EMBED_BATCH, on_batch=None):
    """텍스트 목록 → 단위벡터 목록. task_type 은 RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY.

    429 는 두 가지다 — 요청이 너무 크거나(배치), 너무 잦거나(분당). 앞의 것은 배치를
    절반으로 줄여 다시 시도하고, 뒤의 것은 백오프로 기다린다.
    """
    if provider == "mock":
        vs = [_mock_embed(t, dim) for t in texts]
        if on_batch:
            on_batch(len(vs), len(texts), vs)
        return vs
    from google import genai
    from google.genai import types
    from . import llm
    client = genai.Client()
    def send(part):
        r = client.models.embed_content(
            model=model or EMBED_MODEL, contents=part,
            config=types.EmbedContentConfig(output_dimensionality=dim, task_type=task_type))
        return [_unit(e.values) for e in r.embeddings]

    out, i, size = [], 0, max(1, batch)
    while i < len(texts):
        part = texts[i:i + size]
        for attempt in range(8):
            try:
                vs = send(part)
                out += vs
                i += len(part)
                if on_batch:
                    on_batch(i, len(texts), vs)
                break
            except Exception as e:                       # noqa: BLE001
                if attempt == 7 or not llm._is_transient(e):
                    raise
                if "RESOURCE_EXHAUSTED" in str(e):
                    # 분당 **토큰** 한도다. 배치를 줄여도 다음 분까지는 못 넘기므로 기다린다.
                    if len(part) > 1 and attempt == 0:
                        size = max(1, len(part) // 2)
                        part = texts[i:i + size]
                    time.sleep(EMBED_COOLDOWN + random.random() * 3)
                else:
                    time.sleep(min(2 ** attempt, 32) + random.random())
        else:
            raise RuntimeError(f"임베딩 실패: {i}번째부터 {len(part)}건")
    return out

def embed_build(provider="google", model=None, dim=EMBED_DIM, chunks_path=None, resume=True):
    """corpus/chunks.jsonl → corpus/embeddings.jsonl + meta.

    청크 본문을 그대로 임베딩한다 (RETRIEVAL_DOCUMENT). chunks 를 다시 build 했으면
    이것도 다시 돌려야 한다 — 어긋나면 corpus_warnings 가 잡는다.

    **이어받기**: 무료 티어는 분당 토큰 한도가 낮아 632청크 빌드가 10분 넘게 걸리고
    중간에 끊길 수 있다. 배치가 끝날 때마다 파일에 붙이고, 다시 돌리면 이미 있는
    chunk_id 는 건너뛴다. resume=False 면 처음부터 다시 만든다.
    """
    chunks = load(chunks_path)
    if not chunks:
        print("색인이 비어 있습니다 — `python -m harc.rag build` 를 먼저 실행하세요", file=sys.stderr)
        return
    EMBEDS.parent.mkdir(exist_ok=True)
    done = {}
    if resume and EMBEDS.exists():
        for l in EMBEDS.open(encoding="utf-8"):
            r = json.loads(l)
            done[r["chunk_id"]] = r["v"]
        done = {k: v for k, v in done.items() if len(v) == dim}
    todo = [c for c in chunks if c["chunk_id"] not in done]
    if todo:
        if done:
            print(f"{len(done)}건은 이미 있습니다 — 남은 {len(todo)}건만 만듭니다")
        pending = list(todo)

        def flush(n_done, total, vecs):
            """배치가 끝날 때마다 파일에 붙인다 — 끊겨도 여기까지는 남는다."""
            with EMBEDS.open("a", encoding="utf-8") as f:
                for c, v in zip(pending[:len(vecs)], vecs):
                    f.write(json.dumps({"chunk_id": c["chunk_id"],
                                        "v": [round(x, 6) for x in v]}, ensure_ascii=False) + "\n")
            del pending[:len(vecs)]
            print(f"  {n_done}/{total}", flush=True)

        if not resume or not EMBEDS.exists():
            EMBEDS.write_text("", encoding="utf-8")
        embed_texts([c["text"] for c in todo], "RETRIEVAL_DOCUMENT", provider, model, dim,
                    on_batch=flush)
    # 순서를 chunks.jsonl 에 맞춰 다시 쓴다 (이어받기로 뒤섞였을 수 있다)
    done = {}
    for l in EMBEDS.open(encoding="utf-8"):
        r = json.loads(l)
        done[r["chunk_id"]] = r["v"]
    with EMBEDS.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c["chunk_id"], "v": done[c["chunk_id"]]},
                               ensure_ascii=False) + "\n")
    EMBEDS_META.write_text(json.dumps(
        {"model": "mock" if provider == "mock" else (model or EMBED_MODEL),
         "provider": provider, "dim": dim, "n": len(chunks),
         "chunk_ids_sha1": _ids_sha1(c["chunk_id"] for c in chunks)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(chunks)} embeddings (dim {dim}) → {EMBEDS}")

def _ids_sha1(ids):
    h = hashlib.sha1()
    for i in ids:
        h.update(i.encode()); h.update(b"\0")
    return h.hexdigest()

_EMB = None
def load_embeddings(path=None):
    """{chunk_id: 단위벡터} 와 meta. 없으면 (None, None)."""
    global _EMB
    p = pathlib.Path(path or EMBEDS)
    if not p.exists():
        return None, None
    if _EMB is None or _EMB[2] != p.stat().st_mtime_ns:
        vs = {}
        for l in p.open(encoding="utf-8"):
            r = json.loads(l)
            vs[r["chunk_id"]] = r["v"]
        meta = json.loads(EMBEDS_META.read_text(encoding="utf-8")) if EMBEDS_META.exists() else {}
        _EMB = (vs, meta, p.stat().st_mtime_ns)
    return _EMB[0], _EMB[1]

def embed_warnings(chunks_path=None):
    """임베딩 색인이 chunks.jsonl 을 따라오는지. retriever 가 embed/hybrid 일 때만 의미가 있다."""
    chunks = load(chunks_path)
    vs, meta = load_embeddings()
    if vs is None:
        return ["임베딩 색인 없음: corpus/embeddings.jsonl — "
                "`python -m harc.rag build-embed` 를 실행해야 --retriever embed/hybrid 가 동작합니다"]
    ids = [c["chunk_id"] for c in chunks]
    if set(ids) != set(vs):
        return ["임베딩 색인이 chunks.jsonl 과 어긋납니다 "
                f"(청크 {len(ids)} vs 임베딩 {len(vs)}) — "
                "`python -m harc.rag build-embed` 를 다시 실행하세요"]
    if meta.get("chunk_ids_sha1") and meta["chunk_ids_sha1"] != _ids_sha1(ids):
        return ["임베딩 색인의 청크 순서가 chunks.jsonl 과 다릅니다 — "
                "`python -m harc.rag build-embed` 를 다시 실행하세요"]
    return []

class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.c = chunks
        self.docs = [_tok(x["text"]) for x in chunks]
        self.avg = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(w for d in self.docs for w in set(d))
        N = len(self.docs)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [Counter(d) for d in self.docs]
        self.k1, self.b = k1, b

    def score(self, q, i):
        d, tf = self.docs[i], self.tf[i]
        s = 0.0
        for w in q:
            if w in tf:
                f = tf[w]
                s += self.idf.get(w, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
        return s

def load(path=None):
    p = pathlib.Path(path or CHUNKS)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8")]

RETRIEVERS = ("bm25", "embed", "hybrid")

_INDEX = None
def _bm25_rank(chunks, query, keep):
    """BM25 점수 내림차순 (점수 0 은 버린다). keep: 후보 인덱스 집합."""
    global _INDEX
    if _INDEX is None or len(_INDEX.c) != len(chunks):
        _INDEX = BM25(chunks)
    q = _tok(query)
    scored = [(_INDEX.score(q, i), i) for i in keep]
    scored.sort(reverse=True)
    return [(s, i) for s, i in scored if s > 0]

def _embed_rank(chunks, query, keep, provider):
    """코사인 유사도 내림차순. 벡터가 이미 단위벡터라 내적이 곧 코사인이다."""
    vs, _ = load_embeddings()
    if not vs:
        return []
    qv = embed_texts([query], "RETRIEVAL_QUERY", provider)[0]
    scored = []
    for i in keep:
        v = vs.get(chunks[i]["chunk_id"])
        if v is None:
            continue
        scored.append((sum(a * b for a, b in zip(qv, v)), i))
    scored.sort(reverse=True)
    return scored

def _rrf(*ranked, k=RRF_K, pool=FUSE_POOL):
    """Reciprocal Rank Fusion. 점수 눈금이 다른 검색기를 **순위로만** 섞는다.

    BM25 점수와 코사인은 단위가 달라 정규화가 임의적이다. RRF 는 순위만 쓰므로
    그 임의성을 피한다. 두 검색기가 함께 올린 청크가 위로 온다.
    """
    fused = {}
    for lst in ranked:
        for r, (_, i) in enumerate(lst[:pool], start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + r)
    return sorted(((s, i) for i, s in fused.items()), reverse=True)

def retrieve(query, k=5, exclude_papers=(), retriever="bm25", embed_provider="google"):
    """exclude_papers: 평가 대상 논문 id — 반드시 제외 (정답 누수 방지).

    retriever: bm25(어휘) / embed(의미) / hybrid(RRF 로 둘을 섞음).
    누수 필터는 **세 경우 모두 순위를 매기기 전에** 적용된다.
    """
    if retriever not in RETRIEVERS:
        raise ValueError(f"retriever 는 {RETRIEVERS} 중 하나여야 합니다: {retriever!r}")
    chunks = load()
    if not chunks:
        return []
    ex = set(exclude_papers)
    keep = [i for i in range(len(chunks)) if chunks[i]["paper_id"] not in ex]
    if retriever == "bm25":
        ranked = _bm25_rank(chunks, query, keep)
    elif retriever == "embed":
        ranked = _embed_rank(chunks, query, keep, embed_provider)
    else:
        ranked = _rrf(_bm25_rank(chunks, query, keep),
                      _embed_rank(chunks, query, keep, embed_provider))
    return [dict(chunks[i], score=round(s, 4), retriever=retriever) for s, i in ranked[:k]]

if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["check"]:
        ws = corpus_warnings()
        print("\n".join("WARN: " + w for w in ws) if ws else "corpus 이상 없음")
    elif sys.argv[1:2] == ["build-placebo"]:
        placebo_build()
    elif sys.argv[1:2] == ["build-embed"]:
        embed_build(provider=(sys.argv[2] if len(sys.argv) > 2 else "google"))
    elif sys.argv[1:2] == ["search"]:
        rt = "bm25"
        args = sys.argv[2:]
        if args[:1] and args[0] in RETRIEVERS:
            rt, args = args[0], args[1:]
        for r in retrieve(" ".join(args), retriever=rt):
            print(r["chunk_id"], r["score"], r["text"][:150])
