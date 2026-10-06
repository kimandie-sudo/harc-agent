"""3사 공통 LLM 호출. 모든 호출은 logs/calls.jsonl 에 기록된다."""
import base64, json, os, random, re, time, uuid, datetime, pathlib

LOG = pathlib.Path(__file__).resolve().parent.parent / "logs" / "calls.jsonl"

# 기본 모델 ID는 models.json 에서 읽음. `python run.py --list-models <provider>` 로 확인 후 수정할 것.
_MODELS_FILE = pathlib.Path(__file__).resolve().parent.parent / "models.json"
DEFAULT_MODELS = json.loads(_MODELS_FILE.read_text(encoding="utf-8")) if _MODELS_FILE.exists() else {}
DEFAULT_MODELS.setdefault("mock", "mock")

def _img_b64(path):
    data = pathlib.Path(path).read_bytes()
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    return mime, base64.b64encode(data).decode()

def _extract_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            text = text[s:e + 1]
    return json.loads(text)

def _is_blank(v):
    """None 과 빈 문자열만 '답하지 않음' 으로 본다.

    **빈 리스트·빈 딕셔너리는 답이다.** `affected_conditions: []` 는 "영향받는 조건 없음",
    `not_judgeable: []` 는 "판정 불가인 조건 없음" 이라는 정당한 답이다.
    처음 이 검사를 넣을 때 빈 컨테이너까지 결측으로 쳤다가 결함 에이전트가 무더기로
    실패했다 (2026-09-24, 26B 32런에서 7런 성공 10런 실패).
    0 과 False 도 답이다.
    """
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    return False


def _check_required(obj, required, nullable=()):
    """키가 있어야 하고, **nullable 이 아닌 한 값도 비면 안 된다.**

    예전 판은 `k not in obj` 만 봤다. 모델이 키를 통째로 빠뜨려도, 값을 비워도 통과했다.

    **nullable 이 왜 필요한가.** 프롬프트가 스스로 빈 값을 허용하는 키가 있다.
    aggregate 프롬프트는 출력 형식을 `"process_window": "조건 id 목록 또는 null"` 이라고
    적어 둔다 — 결함 없는 조건 구간이 없으면 null 이 **정답**이다. 그걸 결측으로 치면
    모델은 재시도를 다 태우고 런 전체가 실패한다 (2026-09-25 스모크에서 확인: S4 는
    14조건 전부 언더컷이 있어 flash-lite 가 8회 연속 null 을 냈다).
    프롬프트가 허용한 답을 코드가 거부하면 그건 모델 잘못이 아니라 계약 불일치다.

    그래서 두 가지를 나눈다 — **키가 없는 것**(형식 오류, 재시도)과
    **키는 있고 값이 빈 것**(프롬프트가 허용했으면 답, 아니면 형식 오류).
    """
    absent = [k for k in required if k not in obj]
    blank = [k for k in required if k in obj and k not in nullable and _is_blank(obj[k])]
    if absent or blank:
        raise ValueError(f"필수 키 누락 또는 빈 값: {sorted(absent + blank)}")

# ---------------- provider adapters ----------------
def _call_anthropic(model, system, user, images, temperature, max_tokens):
    import anthropic
    client = anthropic.Anthropic()
    content = []
    for p in images:
        mime, b = _img_b64(p)
        content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b}})
    content.append({"type": "text", "text": user})
    r = client.messages.create(model=model, system=system, max_tokens=max_tokens,
                               temperature=temperature,
                               messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    usage = {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
    return text, usage

def _call_openai(model, system, user, images, temperature, max_tokens, client=None):
    from openai import OpenAI
    client = client or OpenAI()
    content = [{"type": "text", "text": user}]
    for p in images:
        mime, b = _img_b64(p)
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b}"}})
    kw = dict(model=model, response_format={"type": "json_object"},
              messages=[{"role": "system", "content": system}, {"role": "user", "content": content}])
    # 일부 추론 모델은 temperature / max_tokens 를 받지 않음 → 실패 시 제거하고 재시도
    try:
        r = client.chat.completions.create(temperature=temperature, max_completion_tokens=max_tokens, **kw)
    except Exception as e:
        if "temperature" in str(e) or "unsupported" in str(e).lower():
            r = client.chat.completions.create(max_completion_tokens=max_tokens, **kw)
        else:
            raise
    usage = {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}
    return r.choices[0].message.content, usage

def _call_moonshot(model, system, user, images, temperature, max_tokens):
    """Kimi (Moonshot AI). OpenAI 호환. 이미지 입력은 kimi-k3 계열만 지원."""
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["MOONSHOT_API_KEY"], base_url="https://api.moonshot.ai/v1")
    return _call_openai(model, system, user, images, temperature, max_tokens, client=c)

def _call_openrouter(model, system, user, images, temperature, max_tokens):
    """OpenRouter: 키 하나로 여러 회사 모델 (예: moonshotai/kimi-k3, deepseek/..., qwen/...)."""
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    return _call_openai(model, system, user, images, temperature, max_tokens, client=c)

def _is_gemma(model):
    return "gemma" in (model or "").lower()

# Gemma 를 Gemini API 로 부를 때 계정/버전에 따라 거부될 수 있는 옵션들.
# (이 키로 실측했을 때 gemma-4 계열은 둘 다 정상 동작했다. 아래 처리는 거부될 때만 쓰이는 대비책.)
_GEMMA_UNSUPPORTED = ("system_instruction", "systemInstruction", "system instruction",
                      "response_mime_type", "responseMimeType", "response mime", "json mode")

def _call_google(model, system, user, images, temperature, max_tokens):
    from google import genai
    from google.genai import types
    client = genai.Client()  # GEMINI_API_KEY 또는 GOOGLE_API_KEY
    parts = []
    for p in images:
        mime, _ = _img_b64(p)
        parts.append(types.Part.from_bytes(data=pathlib.Path(p).read_bytes(), mime_type=mime))

    # Gemma 계열은 사고(thinking) 토큰이 max_output_tokens 예산을 같이 쓴다.
    # 예산이 작으면 사고만 하다 MAX_TOKENS 로 끊겨 본문이 빈 채로 돌아온다 → 넉넉히 준다.
    if _is_gemma(model):
        max_tokens = max(max_tokens, 8000)

    def _send(fold_system):
        """fold_system=True: system 을 user 앞에 접어 넣고 JSON 모드를 끈 채 보낸다."""
        text = (system + "\n\n" + user) if fold_system else user
        cfg = dict(temperature=temperature, max_output_tokens=max_tokens)
        if not fold_system:
            cfg.update(system_instruction=system, response_mime_type="application/json")
        return client.models.generate_content(
            model=model, contents=parts + [text], config=types.GenerateContentConfig(**cfg))

    try:
        r = _send(False)
    except Exception as e:
        # system 지시문 / JSON 모드를 모델이 거부한 경우(400)에만 접어 넣어 재시도한다.
        # 그 외 오류(429/500 등)는 그대로 올려 보내 공통 백오프 로직이 처리하게 둔다.
        s_err = repr(e)
        if (_is_gemma(model) and ("400" in s_err or "INVALID_ARGUMENT" in s_err)
                and any(k in s_err for k in _GEMMA_UNSUPPORTED)):
            r = _send(True)
        else:
            raise

    text = r.text
    if not text:
        fr = getattr(r.candidates[0], "finish_reason", None) if r.candidates else None
        raise ValueError(f"응답 본문이 비어 있음 (finish_reason={fr})")
    um = r.usage_metadata
    usage = {"in": getattr(um, "prompt_token_count", None), "out": getattr(um, "candidates_token_count", None),
             "thoughts": getattr(um, "thoughts_token_count", None)}
    return text, usage

_MOCK_REQUIRED = []
def _call_mock(model, system, user, images, temperature, max_tokens):
    """API 없이 파이프라인 점검용. 필수 키만 채운 가짜 JSON을 돌려준다."""
    return json.dumps({k: f"(mock {k})" for k in _MOCK_REQUIRED}, ensure_ascii=False), {"in": 0, "out": 0}

ADAPTERS = {"anthropic": _call_anthropic, "openai": _call_openai, "google": _call_google,
            "moonshot": _call_moonshot, "openrouter": _call_openrouter, "mock": _call_mock}

# 일시적 서버/한도 오류 → 지수 백오프 후 재시도 (형식 오류는 즉시 재시도)
_TRANSIENT = ("503", "UNAVAILABLE", "overloaded", "high demand", "429", "RESOURCE_EXHAUSTED",
              "Server disconnected", "RemoteProtocolError", "timed out", "TimeoutError",
              "Connection reset", "502", "504", "500", "INTERNAL")

def _is_transient(e):
    s = repr(e)
    return any(t in s for t in _TRANSIENT)

# 재시도 예산은 provider 별로 다르다.
#  - openrouter 무료 모델: 429(upstream 공유 풀 혼잡)는 일일 한도에 계상되지 않으므로 끈질기게 재시도.
#  - google 무료 티어: 한도가 "모델당 하루 N건"이라 재시도가 그날 예산을 깎지만,
#    500 INTERNAL / 503 이 잦아(특히 gemma-4-31b-it) 3회로는 런이 통째로 날아간다 → 5회.
#  - 2026-09-23: google 이 503 UNAVAILABLE("high demand") 파도를 보내 48런 배치에서
#    첫 런이 통째로 실패했다. 예산 5회 x 백오프 상한 20초면 총 대기가 100초뿐이라
#    몇 분짜리 혼잡 구간을 못 넘긴다. 예산을 8로, 상한을 BACKOFF_CAP 으로 올렸다.
RETRY_BUDGET = {"openrouter": 25, "moonshot": 8, "google": 8, "anthropic": 5, "openai": 5, "mock": 0}

# multi 의 결함 에이전트를 동시에 몇 개까지 부를지. 없으면 전부(9개) 동시에.
# google 무료 티어는 분당 입력 토큰 한도(gemma-4-26b 16,000/분)가 있어, 9개를 한꺼번에 보내면 429 가 나고
# 재시도가 하루 요청 한도를 깎는다 — 2026-10-02 v3 기준선에서 26B 호출의 약 절반이 이 429 였다.
# 결과에는 영향이 없다(각 에이전트는 서로의 출력을 보지 않는다). 속도와 할당량만 바뀐다.
MAX_PARALLEL = {"google": 3}
BACKOFF_CAP = 60      # 일시 오류 재시도 대기 상한(초). 서버 혼잡은 분 단위로 간다.

def call_json(agent, provider, system, user, required=(), model=None, images=(),
              temperature=0.0, max_tokens=4000, retries=None, run_id=None, nullable=()):
    if retries is None:
        retries = RETRY_BUDGET.get(provider, 5)
    model = model or DEFAULT_MODELS[provider]
    _MOCK_REQUIRED[:] = list(required)
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.time()
        rec = {"id": uuid.uuid4().hex[:10], "run_id": run_id, "agent": agent, "provider": provider,
               "model": model, "temperature": temperature, "attempt": attempt,
               "time": datetime.datetime.now().isoformat(timespec="seconds"),
               "images": [str(p) for p in images], "system": system, "user": user}
        try:
            text, usage = ADAPTERS[provider](model, system, user, list(images), temperature, max_tokens)
            rec.update(raw=text, usage=usage, sec=round(time.time() - t0, 2))
            obj = _extract_json(text)
            _check_required(obj, required, nullable)
            rec["ok"] = True
            _log(rec)
            return obj
        except Exception as e:
            last_err = e
            transient = _is_transient(e)
            rec.update(ok=False, error=repr(e)[:500], sec=round(time.time() - t0, 2),
                       transient=transient)
            _log(rec)
            if attempt == retries:
                break
            if transient:
                # 서버 과부하·한도: 프롬프트는 그대로 두고 백오프 후 재시도.
                # 무료 티어(google AI Studio 공유 풀)는 429가 잦아 재시도 예산을 크게 잡는다.
                # jitter 는 병렬 결함 에이전트 3개가 같은 순간에 몰려 재시도하는 것을 막는다.
                time.sleep(min(3 * 2 ** attempt, BACKOFF_CAP) * (0.5 + random.random()))
            else:
                user = user + "\n\n(이전 응답이 형식 오류였음: JSON 객체 하나만, 필수 키 모두 포함해 다시 답할 것.)"
    raise RuntimeError(f"[{agent}/{provider}] 실패: {last_err}")

def _log(rec):
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def list_models(provider):
    if provider == "anthropic":
        import anthropic
        return [m.id for m in anthropic.Anthropic().models.list()]
    if provider == "openai":
        from openai import OpenAI
        return sorted(m.id for m in OpenAI().models.list())
    if provider == "google":
        from google import genai
        cl = genai.Client()
        return [m.name for m in cl.models.list()]
    if provider in ("moonshot", "openrouter"):
        from openai import OpenAI
        env, url = {"moonshot": ("MOONSHOT_API_KEY", "https://api.moonshot.ai/v1"),
                    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1")}[provider]
        return sorted(m.id for m in OpenAI(api_key=os.environ[env], base_url=url).models.list())
    return ["mock"]
