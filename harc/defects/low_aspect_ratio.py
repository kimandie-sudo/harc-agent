from .base import (Defect, direction_block, verdict_vs_first, trend, show_trend,
                   LOWER_IS_WORSE, SAME)


class LowAspectRatio(Defect):
    """종횡비 부족 — 깊이/폭이 설계 요구에 못 미치는 것.

    S4 의 정답(저자 최적)은 "설계 요구 AR > 10 을 지키면서 균형" 인데, 이 모듈 전에는 종횡비가 프롬프트에
    들어가지 않아 모델이 그 요구를 볼 수 없었다 (2026-10-02 분석). 시리즈에 target_aspect_ratio 가 있으면
    조건마다 요구 충족 여부를 함께 적는다.
    """
    name = "low_aspect_ratio"
    label = "종횡비 부족"
    requires = [("aspect_ratio",)]
    definition = "식각 깊이 대비 폭(종횡비)이 설계 요구에 못 미치는 것. 고종횡비 식각에서는 다른 결함을 줄이는 조건이 종횡비를 깎는 일이 흔하다."
    metric = "aspect_ratio"
    metric_definition = "종횡비 = 깊이/폭 (작을수록 심함)"
    direction = LOWER_IS_WORSE
    direction_note = "값이 클수록 더 깊게 파임"

    def rule(self, series, judgeable):
        target = series.get("target_aspect_ratio")
        rows, ref = [], None
        for c in series["conditions"]:
            j = judgeable.get(c["id"], {})
            if not j.get("ok"):
                rows.append({"condition": c["id"], "level": c.get("level"),
                             "status": "판정 불가", "reason": j.get("reason")})
                continue
            v = c["aspect_ratio"]
            row = {"condition": c["id"], "level": c.get("level"), "aspect_ratio": v}
            if target is not None:
                row["meets_target"] = v > target
            if ref is None:
                ref = v
            row["change_vs_first"] = verdict_vs_first(v, ref, self.direction, tol=0.05)
            row["significant_vs_first"] = row["change_vs_first"] != SAME
            rows.append(row)
        out = {"defect": self.name, **self.meta(),
               "change_vs_first_basis": "첫 조건 대비 종횡비 변화율. 5% 이내는 '변화 없음'",
               "rows": rows}
        if target is not None:
            out["target_aspect_ratio"] = target
            out["meets_target_conditions"] = [r["condition"] for r in rows if r.get("meets_target")]
        out.update(direction_block([(r["condition"], r.get(self.metric)) for r in rows],
                                   self.direction, self.direction_note))
        if show_trend(series):
            out["trend_aspect_ratio"] = trend([r.get("aspect_ratio") for r in rows])
        return out
