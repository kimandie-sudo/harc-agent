from .base import (Defect, THRESHOLDS, trend, show_trend, direction_block, verdict_vs_first,
                   HIGHER_IS_WORSE, SAME)


class CdDeviation(Defect):
    """설계 CD 대비 편차 — 넓어짐과 좁아짐 **둘 다** 나쁘다.

    기준은 시리즈의 nominal_cd_nm(마스크 개구 또는 설계 CD)이다. 값은 cd_nm(측정 위치가 원문에
    명시되지 않은 홀 직경)이 있으면 그것을, 없으면 top_cd 를 쓴다. 방향이 양쪽이라 심각도는
    **절댓값**으로 매기고, 부호가 있는 값은 행에 따로 남긴다 — 넓어졌는지 좁아졌는지가 원인을 가른다.
    """
    name = "cd_deviation"
    label = "CD 편차(넓어짐·좁아짐)"
    requires = [("cd_nm",), ("top_cd",)]
    definition = "식각 후 패턴 CD 가 설계(마스크 개구) CD 에서 벗어나는 것. 넓어짐은 측면 과식각·마스크 침식, 좁아짐은 폴리머 과증착과 관련된다."
    metric = "cd_dev_abs_pct"
    metric_definition = "|측정CD/설계CD − 1| × 100 % (클수록 심함). 부호 있는 값은 cd_dev_pct"
    direction = HIGHER_IS_WORSE
    direction_note = "0 에 가까울수록 설계 CD 에 맞음 (넓어짐·좁아짐 모두 나쁨)"

    def rule(self, series, judgeable):
        nominal = series.get("nominal_cd_nm")
        rows, ref = [], None
        for c in series["conditions"]:
            j = judgeable.get(c["id"], {})
            cd = c.get("cd_nm", c.get("top_cd"))
            if not j.get("ok") or not nominal or cd is None:
                rows.append({"condition": c["id"], "level": c.get("level"), "status": "판정 불가",
                             "reason": j.get("reason") or "시리즈에 설계 CD(nominal_cd_nm)가 없음"})
                continue
            dev = (cd / nominal - 1) * 100
            row = {"condition": c["id"], "level": c.get("level"), "cd_nm": cd, "nominal_cd_nm": nominal,
                   "cd_dev_pct": round(dev, 1), "cd_dev_abs_pct": round(abs(dev), 1),
                   "cd_dev_side": "넓어짐" if dev > 0 else ("좁아짐" if dev < 0 else "일치")}
            if ref is None:
                ref = abs(dev)
            row["change_vs_first"] = verdict_vs_first(abs(dev), ref, self.direction,
                                                      tol=THRESHOLDS["cd_change"] * 100, relative=False)
            row["significant_vs_first"] = row["change_vs_first"] != SAME
            rows.append(row)
        out = {"defect": self.name, **self.meta(), "nominal_cd_nm": nominal,
               "threshold_pct": THRESHOLDS["cd_change"] * 100,
               "change_vs_first_basis": "첫 조건 대비 |편차| 의 차(%p). 3%p 이내는 '변화 없음'",
               "rows": rows}
        out.update(direction_block([(r["condition"], r.get(self.metric)) for r in rows],
                                   self.direction, self.direction_note))
        if show_trend(series):
            out["trend_cd_dev"] = trend([r.get("cd_dev_pct") for r in rows])
        return out
