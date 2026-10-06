from .base import (Defect, THRESHOLDS, trend, show_trend, direction_block, verdict_vs_first,
                   HIGHER_IS_WORSE, SAME)


class Undercut(Defect):
    """마스크 아래가 옆으로 파이는 현상 (mask undercut).

    보잉(bowing)과 같은 계열 — 둘 다 측면 과식각이다. 그러나 **같은 지표가 아니다**:
      bowing   최대CD/상단CD 비 (무차원). 구멍 **안쪽** 이 부푼 정도
      undercut (W2 − W1)/2 (nm). 마스크 **가장자리** 가 옆으로 파인 절대 폭
    지표가 다르므로 모듈을 나눈다. 비로 환산할 수 있는 시리즈라면 bowing 을 쓸 것.
    """
    name = "undercut"
    label = "언더컷(마스크 아래 측면 파임)"
    requires = [("undercut_nm",)]
    definition = "마스크 가장자리 아래가 옆으로 파이는 현상. 패턴 폭이 설계보다 넓어진다."
    metric = "undercut_nm"
    metric_definition = "(최대폭 − 개구폭)/2, 단위 nm (클수록 심함)"
    direction = HIGHER_IS_WORSE
    direction_note = "값이 작을수록 마스크 아래가 덜 파임"

    def rule(self, series, judgeable):
        rows, ref = [], None
        for c in series["conditions"]:
            j = judgeable.get(c["id"], {})
            if not j.get("ok"):
                rows.append({"condition": c["id"], "level": c.get("level"),
                             "status": "판정 불가", "reason": j.get("reason")})
                continue
            u = c["undercut_nm"]
            row = {"condition": c["id"], "level": c.get("level"), "undercut_nm": u}
            if ref is None:
                ref = u
            row["change_vs_first_pct"] = round((u / ref - 1) * 100, 1) if ref else None
            row["change_vs_first"] = verdict_vs_first(u, ref, self.direction,
                                                      tol=THRESHOLDS["cd_change"])
            row["significant_vs_first"] = row["change_vs_first"] != SAME
            rows.append(row)
        out = {"defect": self.name, **self.meta(),
               "threshold_pct": THRESHOLDS["cd_change"] * 100,
               "change_vs_first_basis": "첫 조건 대비 언더컷 비율 변화. 3% 이내는 '변화 없음'",
               "rows": rows}
        out.update(direction_block([(r["condition"], r.get(self.metric)) for r in rows],
                                   self.direction, self.direction_note))
        if show_trend(series):
            out["trend_undercut"] = trend([r.get("undercut_nm") for r in rows])
        return out
