from .base import Defect, direction_block, verdict_vs_first, TRUE_IS_WORSE, SAME

class EtchStop(Defect):
    name = "etch_stop"
    label = "식각 중단"
    requires = [("etch_stop",)]
    definition = "목표 깊이에 도달하기 전에 식각이 멈추는 현상."
    metric = "etch_stop"
    metric_definition = "불리언 — true 이면 중단"
    direction = TRUE_IS_WORSE
    direction_note = "중단(true)이면 나쁨"

    def rule(self, series, judgeable):
        rows, ref = [], None
        for c in series["conditions"]:
            if not judgeable.get(c["id"], {}).get("ok"):
                rows.append({"condition": c["id"], "level": c.get("level"), "status": "판정 불가",
                             "reason": judgeable.get(c["id"], {}).get("reason")})
                continue
            row = {"condition": c["id"], "level": c.get("level"), "etch_stop": c["etch_stop"]}
            if ref is None:
                ref = c["etch_stop"]
            row["change_vs_first"] = verdict_vs_first(c["etch_stop"], ref, self.direction,
                                                     relative=False)
            row["significant_vs_first"] = row["change_vs_first"] != SAME
            rows.append(row)
        onset = next((r["condition"] for r in rows if r.get("etch_stop")), None)
        out = {"defect": self.name, **self.meta(),
               "change_vs_first_basis": "첫 조건 대비 상태 변화 (정상→중단은 '악화')",
               "rows": rows, "onset": onset}
        out.update(direction_block([(r["condition"], r.get(self.metric)) for r in rows],
                                   self.direction, self.direction_note))
        return out
