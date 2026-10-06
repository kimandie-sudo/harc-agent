"""[1] 데이터 점검: 완결성 + 해상도 판별 한계 + (선택) 이미지 품질."""
import pathlib
from .defects.registry import ALL
from .defects.base import THRESHOLDS
from .llm import call_json

EDGE_PX = 2  # 가장자리 위치 불확실성(px). 양쪽 가장자리 합산 근사

def resolution_limit(series):
    px = series.get("px_nm")
    out = {"px_nm": px, "edge_uncertainty_px": EDGE_PX, "per_condition": {}}
    if not px:
        out["note"] = "px 크기 정보 없음 → 이미지 기반 판별 한계 계산 불가"
        return out
    for c in series["conditions"]:
        cd = c.get("top_cd") or series.get("nominal_cd_nm")
        if not cd:
            out["per_condition"][c["id"]] = {"min_detectable_pct": None, "reason": "CD 기준값 없음"}
            continue
        m = EDGE_PX * px / cd
        out["per_condition"][c["id"]] = {
            "cd_ref_nm": cd, "min_detectable_pct": round(m * 100, 1),
            "can_resolve_cd_threshold": m <= THRESHOLDS["cd_change"]}
    return out

def check(series, provider=None, model=None, use_images=False, run_id=None, base_dir=".",
          temperature=0.0):
    res = resolution_limit(series)
    from_paper = series.get("value_source") == "paper"
    table, gaps = {}, []
    for d in ALL:
        table[d.name] = {}
        for c in series["conditions"]:
            ok, used = d.available(c)
            if ok:
                table[d.name][c["id"]] = {"ok": True, "uses": list(used)}
                continue
            missing = sorted({k for req in d.requires for k in req if c.get(k) is None})
            reason = "값 없음: " + ", ".join(missing)
            img = c.get("image")
            if c.get("etch_stop") and ("bottom_cd" in missing):
                reason += " (식각 중단으로 하단 CD 정의 안 됨 — 결측이 아니라 결과)"
                img = None  # 이미지로도 보충 불가
            has_img = img and (pathlib.Path(base_dir) / img).exists()
            lim = res["per_condition"].get(c["id"], {}).get("min_detectable_pct")
            if img:
                hint = f"이미지에서 harc_measure로 {', '.join(missing)} 측정 시 판정 가능"
                if lim is not None:
                    hint += f" (이 해상도에서 판별 가능한 최소 변화 약 {lim}%)"
                reason += f" / 보충 가능: {hint}" + ("" if has_img else " [이미지 파일 없음]")
            table[d.name][c["id"]] = {"ok": False, "reason": reason, "missing": missing}
            gaps.append({"defect": d.name, "condition": c["id"], "missing": missing})
    report = {
        "judgeable": table,
        "resolution": res,
        "value_source_note": ("계측·논문 표기값 사용 — 해상도 한계는 값을 이미지에서 새로 잴 때만 적용"
                              if from_paper else "이미지 측정값 — 해상도 한계 미만의 차이는 '변화 없음'으로 취급"),
        "gaps": gaps,
    }
    if use_images and provider:
        report["image_quality"] = image_quality(series, provider, model, run_id, base_dir, temperature)
    return report

IQ_SYSTEM = """당신은 단면 SEM 이미지의 품질만 점검합니다. 공정 판단은 하지 마세요.
각 이미지에 대해 JSON 하나로 답하세요:
{"items": [{"condition": id, "bottom_visible": true/false, "cleave_through_center": "yes/no/unclear",
 "artifacts": ["charging", "tilt", "blur" 등 또는 빈 목록], "usable_for": ["top_cd","max_cd","bottom_cd","depth","sidewall"]}]}"""

def image_quality(series, provider, model, run_id, base_dir, temperature=0.0):
    imgs, ids = [], []
    for c in series["conditions"]:
        p = pathlib.Path(base_dir) / c.get("image", "")
        if c.get("image") and p.exists():
            imgs.append(p); ids.append(c["id"])
    if not imgs:
        return {"skipped": "이미지 파일 없음"}
    user = "이미지 순서대로 조건 id: " + ", ".join(ids) + f". 1 px = {series.get('px_nm')} nm."
    return call_json("checker_image", provider, IQ_SYSTEM, user, required=["items"],
                     model=model, images=imgs, run_id=run_id, temperature=temperature)
