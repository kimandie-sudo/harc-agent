from .base import (Defect, THRESHOLDS, trend, show_trend, direction_block, verdict_vs_first,
                   HIGHER_IS_WORSE, SAME)

class Bowing(Defect):
    name = "bowing"
    label = "보잉"
    requires = [("top_cd", "max_cd")]
    definition = "구멍 중간 깊이에서 폭이 입구(상단 CD)보다 넓어지는 현상."
    metric = "bow_index_pct"
    metric_definition = "(최대CD/상단CD - 1) × 100 % (클수록 심함)"
    direction = HIGHER_IS_WORSE
    direction_note = "값이 작을수록 측벽이 곧음"

    def rule(self, series, judgeable):
        rows, ref = [], None
        tol_pp = THRESHOLDS["cd_change"] * 100   # 보잉지수는 % 값 → 퍼센트포인트 차이로 비교
        for c in series["conditions"]:
            if not judgeable.get(c["id"], {}).get("ok"):
                rows.append({"condition": c["id"], "level": c.get("level"), "status": "판정 불가",
                             "reason": judgeable.get(c["id"], {}).get("reason")})
                continue
            bi = c["max_cd"] / c["top_cd"] - 1
            row = {"condition": c["id"], "level": c.get("level"),
                   "top_cd": c["top_cd"], "max_cd": c["max_cd"],
                   "max_cd_depth": c.get("max_cd_depth"),
                   "bow_index_pct": round(bi * 100, 1),
                   "bowing": bi > THRESHOLDS["bow_index"]}
            if ref is None:
                ref = row["bow_index_pct"]
            row["change_vs_first_pp"] = round(row["bow_index_pct"] - ref, 1)
            row["change_vs_first"] = verdict_vs_first(row["bow_index_pct"], ref, self.direction,
                                                      tol=tol_pp, relative=False)
            row["significant_vs_first"] = row["change_vs_first"] != SAME
            rows.append(row)
        out = {"defect": self.name, **self.meta(), "threshold_pct": THRESHOLDS["bow_index"] * 100,
               "change_vs_first_basis": f"첫 조건 대비 퍼센트포인트 차이. {tol_pp:g}pp 이내는 '변화 없음'",
               "rows": rows}
        if show_trend(series):      # 요인·시퀀스 설계에서는 조건 순서가 추세를 뜻하지 않는다
            out["trend_bow_index"] = trend([r.get("bow_index_pct") for r in rows])
        out.update(direction_block([(r["condition"], r.get(self.metric)) for r in rows],
                                   self.direction, self.direction_note))
        return out
