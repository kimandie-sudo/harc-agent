from .base import (Defect, THRESHOLDS, trend, show_trend, direction_block, verdict_vs_first,
                   HIGHER_IS_WORSE, SAME)


class SidewallAngle(Defect):
    """측벽이 수직에서 벗어난 각도 (inverse sidewall angle).

    테이퍼(taper)와 같은 계열이지만 **지표가 다르다**:
      taper          상단CD/하단CD 비 (무차원)
      sidewall_angle 수직으로부터 벗어난 각도, 단위 도(°)
    0°가 완전 수직이고 클수록 기울었다. 비로 환산하려면 깊이가 필요한데
    그 값이 없는 논문이 많아 각도를 그대로 둔다.
    """
    name = "sidewall_angle"
    label = "측벽각(수직 이탈)"
    requires = [("sidewall_angle_deg",)]
    definition = "측벽이 수직에서 벗어난 각도. 0°가 완전 수직이며 클수록 기울어져 있다."
    metric = "sidewall_angle_deg"
    metric_definition = "수직으로부터의 이탈 각도, 단위 도 (클수록 심함)"
    direction = HIGHER_IS_WORSE
    direction_note = "값이 작을수록 수직"

    def rule(self, series, judgeable):
        rows, ref = [], None
        for c in series["conditions"]:
            j = judgeable.get(c["id"], {})
            if not j.get("ok"):
                rows.append({"condition": c["id"], "level": c.get("level"),
                             "status": "판정 불가", "reason": j.get("reason")})
                continue
            a = c["sidewall_angle_deg"]
            row = {"condition": c["id"], "level": c.get("level"), "sidewall_angle_deg": a}
            if ref is None:
                ref = a
            # 각도는 0 에 가까울 수 있어 비율 변화가 폭발한다 → 절대 차(도)로 비교한다.
            row["change_vs_first_deg"] = round(a - ref, 2)
            row["change_vs_first"] = verdict_vs_first(a, ref, self.direction,
                                                      tol=THRESHOLDS["sidewall_angle_deg"],
                                                      relative=False)
            row["significant_vs_first"] = row["change_vs_first"] != SAME
            rows.append(row)
        out = {"defect": self.name, **self.meta(),
               "threshold_deg": THRESHOLDS["sidewall_angle_deg"],
               "change_vs_first_basis": "첫 조건 대비 각도 차(도). 0.5° 이내는 '변화 없음'. "
                                        "각도는 0 에 가까울 수 있어 비율이 아니라 절대 차로 잰다",
               "rows": rows}
        out.update(direction_block([(r["condition"], r.get(self.metric)) for r in rows],
                                   self.direction, self.direction_note))
        if show_trend(series):
            out["trend_sidewall_angle"] = trend([r.get("sidewall_angle_deg") for r in rows])
        return out
