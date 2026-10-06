"""사용 예
  python run.py --series S1 --provider mock
  python run.py --series S1 --provider anthropic --mode multi
  python run.py --series S1 S2 --provider all --mode multi single --rag off on --reps 3
  python run.py --list-models openai
"""
import argparse, itertools, json, pathlib, re, sys
from harc import PIPELINE_VERSION
from harc.pipeline import run
from harc.llm import list_models, DEFAULT_MODELS

ROOT = pathlib.Path(__file__).resolve().parent
PROVIDERS = ["anthropic", "openai", "google", "moonshot"]  # --provider all 대상
MAX_CONSECUTIVE_FAIL = 3   # 연속 실패가 이만큼이면 배치를 멈춘다

def _slug(model):
    """모델 ID를 파일명에 쓸 수 있게 줄인다. gemma-4-31b-it → gemma-4-31b-it"""
    return re.sub(r"[^A-Za-z0-9.-]+", "-", model.split("/")[-1]).strip("-")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", default=["S1"])
    ap.add_argument("--provider", nargs="+", default=["mock"], help="anthropic openai google mock 또는 all")
    ap.add_argument("--model", default=None, help="모델 ID 직접 지정 (provider 하나일 때)")
    ap.add_argument("--mode", nargs="+", default=["multi"], choices=["multi", "single"])
    ap.add_argument("--rag", nargs="+", default=["off"], choices=["off", "on"])
    ap.add_argument("--images", nargs="+", default=["off"], choices=["off", "on"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-parallel", type=int, default=None, metavar="N",
                    help="multi 의 결함 에이전트 동시 호출 수. 기본은 제공사별(google 3, 나머지 9). 결과는 같고 속도·할당량만 바뀐다")
    ap.add_argument("--skip-existing", action="store_true",
                    help="outdir 에 같은 조건·반복 번호의 결과가 이미 있으면 건너뛴다 (끊긴 배치 재개용)")
    ap.add_argument("--temperature", nargs="+", type=float, default=[0.0], metavar="T",
                    help="샘플링 온도. 기본 0.0 은 사실상 결정론이라 --reps 를 올려도 "
                         "서버측 비결정성만 재게 된다. 반복으로 변동을 보려면 0 보다 크게 줄 것")
    ap.add_argument("--rule-hints", nargs="+", default=["on"], choices=["on", "off"],
                    help="규칙 판정의 방향 힌트(direction·worst/best_condition·개선/악화)를 "
                         "프롬프트에 넣을지. off 면 모델이 스스로 방향을 판단해야 한다")
    ap.add_argument("--retriever", nargs="+", default=["bm25"],
                    choices=["bm25", "embed", "hybrid"],
                    help="검색 방식. bm25=어휘 겹침(기본), embed=임베딩 의미 유사도, "
                         "hybrid=둘을 RRF 로 섞음. embed/hybrid 는 "
                         "`python -m harc.rag build-embed` 로 임베딩 색인을 먼저 만들어야 한다")
    ap.add_argument("--chunk-source", nargs="+", default=["rag"],
                    choices=["rag", "placebo", "planted"],
                    help="프롬프트에 넣을 청크의 출처. rag=검색 결과(기본), "
                         "placebo=길이를 맞춘 무관한 청크(literature/xai), "
                         "planted=최하위 청크 앞에 '이 시리즈에서는 {틀린 조건}이 공정 창' 문장을 "
                         "붙인 것(양성 대조군, data/planted.json). 기제 실험 조건 "
                         "(DESIGN_기제실험.md). placebo 는 `python -m harc.rag build-placebo` 를 먼저, "
                         "planted 는 --anonymize-chunks on 과 함께만")
    ap.add_argument("--anonymize-chunks", nargs="+", default=["off"], choices=["off", "on"],
                    help="프롬프트의 chunk_id 를 D1…D6 으로 바꾼다. 논문 이름이 위약·심은 청크를 "
                         "드러내지 않게 하려는 것. 대응표는 런 JSON 의 chunk_alias")
    ap.add_argument("--chunk-chars", nargs="+", type=int, default=[0], metavar="N",
                    help="RAG 청크를 N 자로 일부러 자른다 (0=자르지 않음). 절단 효과를 재는 실험 조건")
    ap.add_argument("--cause-hedge", nargs="+", default=["on"], choices=["on", "off"],
                    help="multi 의 cause 프롬프트에 인용 면책 조항(basis=general_knowledge)을 둘지. "
                         "기본 on 이 v2 까지의 판. off 는 single 과 같은 수준의 인용 지시만 준다")
    ap.add_argument("--tag", default=None,
                    help="출력 파일명에 끼워 넣을 꼬리표. 같은 provider 를 다른 모델로 돌린 결과를 구분할 때 쓴다.")
    ap.add_argument("--outdir", default="outputs",
                    help="결과 JSON 을 쓸 디렉터리 (기본 outputs). v1 파일럿과 섞이지 않게 나눌 때 씁니다.")
    ap.add_argument("--list-models")
    a = ap.parse_args()

    if a.list_models:
        print("\n".join(list_models(a.list_models))); return

    provs = PROVIDERS if a.provider == ["all"] else a.provider
    outdir = ROOT / a.outdir; outdir.mkdir(parents=True, exist_ok=True)
    print(f"pipeline_version={PIPELINE_VERSION} → {outdir}")
    fails = 0
    for sid, prov, mode, rg, im, hedge, cut, temp, hints, retr, csrc, anon, rep in itertools.product(
            a.series, provs, a.mode, a.rag, a.images, a.cause_hedge, a.chunk_chars,
            a.temperature, a.rule_hints, a.retriever, a.chunk_source, a.anonymize_chunks,
            range(a.reps)):
        series = json.loads((ROOT / "data" / "series" / f"{sid}.json").read_text(encoding="utf-8"))
        slug = a.tag or (_slug(a.model) if a.model else None)
        tag = (f"{sid}_{prov}" + (f"-{slug}" if slug else "")
               + f"_{mode}_rag-{rg}_img-{im}"
               + ("" if hedge == "on" else "_hedge-off")   # 기본값은 파일명을 바꾸지 않는다
               + (f"_cut{cut}" if cut else "")
               + (f"_t{temp:g}" if temp else "")      # 기본값 0.0 은 파일명을 바꾸지 않는다
               + ("" if hints == "on" else "_nohints")
               + ("" if retr == "bm25" else f"_{retr}")   # 기본값은 파일명을 바꾸지 않는다
               + ("" if csrc == "rag" else f"_{csrc}")
               + ("" if anon == "off" else "_anon")
               + f"_r{rep+1}")
        if a.skip_existing and any(outdir.glob(f"{tag}_*.json")):
            print("SKIP", tag)
            continue
        try:
            res = run(series, provider=prov, model=a.model, mode=mode, use_rag=rg == "on",
                      use_images=im == "on", base_dir=str(ROOT / "data"),
                      cause_hedge=hedge == "on", chunk_chars=cut or None, temperature=temp,
                      rule_hints=hints == "on", retriever=retr, chunk_source=csrc,
                      anonymize_chunks=anon == "on", max_parallel=a.max_parallel)
            res["model"] = a.model or DEFAULT_MODELS.get(prov)
            p = outdir / f"{tag}_{res['run_id']}.json"
            p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print("OK  ", tag, "→", p.name)
            fails = 0
        except Exception as e:
            print("FAIL", tag, e, file=sys.stderr)
            fails += 1
            # 실패가 쌓이는데 계속 돌면 할당량만 탄다 (2026-09-24 에 26B 를 645/500 까지 태웠다)
            if fails >= MAX_CONSECUTIVE_FAIL:
                print(f"연속 실패 {fails}회 — 멈춘다. --skip-existing 으로 재개할 것", file=sys.stderr)
                sys.exit(1)

if __name__ == "__main__":
    main()
