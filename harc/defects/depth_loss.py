from .base import (Defect, THRESHOLDS, trend, show_trend, direction_block, verdict_vs_first,
                   LOWER_IS_WORSE, SAME)


class DepthLoss(Defect):
    """식각 깊이 감소 — 같은 사이클·시간에서 덜 파이는 것.

    etch_stop(불리언)과 같은 계열이지만 **멈추지 않고 덜 파이는** 경우를 잰다. 프로파일을 고치는
    조건(패시베이션 강화, ICP 파워 증가)이 깊이를 깎는 트레이드오프가 흔해서, 이 값이 없으면
    프로파일 지표만 보고 판단하게 된다. 조건 사이의 식각 시간이 같을 때만 비교가 의미 있다.
    """
    name = "depth_loss"
    label = "식각 깊이 감소"
    requires = [("etch_depth_um",)]
    definition = ("같은 식각 시간(사이클)에서 식각 깊이가 줄어드는 현상. 홀이 좁고 깊을수록 이온·라디칼이 "
                  "바닥에 덜 닿아 생기고(ARDE), 측벽·바닥 폴리머가 두꺼워져도 생긴다.")
    metric = "etch_depth_um"
    metric_definition = "식각 깊이, 단위 µm (작을수록 심함)"
    direction = LOWER_IS_WORSE
    direction_note = "값이 클수록 같은 시간에 깊이 파임"

    def rule(self, series, judgeable):
        rows, ref = [], None
        tol = THRESHOLDS["depth_change"]
        for c in series["conditions"]:
            j = judgeable.get(c["id"], {})
            if not j.get("ok"):
                rows.append({"condition": c["id"], "level": c.get("level"),
                             "status": "판정 불가", "reason": j.get("reason")})
                continue
            v = c["etch_depth_um"]
            row = {"condition": c["id"], "level": c.get("level"), "etch_depth_um": v}
            if ref is None:
                ref = v
            row["change_vs_first_pct"] = round((v / ref - 1) * 100, 1)
            row["change_vs_first"] = verdict_vs_first(v, ref, self.direction, tol=tol)
            row["significant_vs_first"] = row["change_vs_first"] != SAME
            rows.append(row)
        out = {"defect": self.name, **self.meta(), "threshold_pct": tol * 100,
               "change_vs_first_basis": f"첫 조건 대비 깊이 변화율. {tol * 100:.0f}% 이내는 '변화 없음'",
               "rows": rows}
        out.update(direction_block([(r["condition"], r.get(self.metric)) for r in rows],
                                   self.direction, self.direction_note))
        if show_trend(series):
            out["trend_depth"] = trend([r.get("etch_depth_um") for r in rows])
        return out
