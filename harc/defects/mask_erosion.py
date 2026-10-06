from .base import (Defect, THRESHOLDS, trend, show_trend, direction_block, verdict_vs_first,
                   HIGHER_IS_WORSE, SAME)


class MaskErosion(Defect):
    """마스크 침식 — 마스크가 식각되어 없어지는 속도.

    지표는 **마스크 식각률**(nm/min)이다. 선택비(피식각막/마스크)가 더 흔한 표현이지만,
    마스크가 깎이지 않으면 선택비가 무한대가 되어 비교할 수 없다. 식각률은 음수(마스크 위 증착)까지
    한 줄에 놓인다. 피식각막 식각률(etch_rate_nm_min)이 함께 있으면 선택비도 같이 보여 준다.
    """
    name = "mask_erosion"
    label = "마스크 침식"
    requires = [("mask_etch_rate_nm_min",)]
    definition = ("식각 중 마스크가 깎여 얇아지거나 모서리가 무너지는 현상. 심하면 패턴이 넓어지고 선택비가 떨어진다. "
                  "마스크 식각률이 음수면 마스크 위에 막이 쌓이고 있다는 뜻이다.")
    metric = "mask_etch_rate_nm_min"
    metric_definition = "마스크 식각률, 단위 nm/min (클수록 심함)"
    direction = HIGHER_IS_WORSE
    direction_note = "값이 작을수록 마스크가 덜 깎임 (음수는 증착)"

    def rule(self, series, judgeable):
        rows, ref = [], None
        tol = THRESHOLDS["mask_etch_rate_nm_min"]
        for c in series["conditions"]:
            j = judgeable.get(c["id"], {})
            if not j.get("ok"):
                rows.append({"condition": c["id"], "level": c.get("level"),
                             "status": "판정 불가", "reason": j.get("reason")})
                continue
            m = c["mask_etch_rate_nm_min"]
            row = {"condition": c["id"], "level": c.get("level"), "mask_etch_rate_nm_min": m}
            er = c.get("etch_rate_nm_min")
            if er is not None:
                row["selectivity"] = round(er / m, 1) if m > 0 else "∞ (마스크가 깎이지 않음)"
            if ref is None:
                ref = m
            # 0·음수를 지나므로 비율이 아니라 절대 차로 잰다
            row["change_vs_first_nm_min"] = round(m - ref, 1)
            row["change_vs_first"] = verdict_vs_first(m, ref, self.direction, tol=tol, relative=False)
            row["significant_vs_first"] = row["change_vs_first"] != SAME
            rows.append(row)
        out = {"defect": self.name, **self.meta(),
               "threshold_nm_min": tol,
               "change_vs_first_basis": f"첫 조건 대비 마스크 식각률의 절대 차. {tol} nm/min 이내는 '변화 없음'",
               "rows": rows}
        out.update(direction_block([(r["condition"], r.get(self.metric)) for r in rows],
                                   self.direction, self.direction_note))
        if show_trend(series):
            out["trend_mask_etch_rate"] = trend([r.get("mask_etch_rate_nm_min") for r in rows])
        return out
