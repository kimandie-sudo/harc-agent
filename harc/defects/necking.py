from .base import (Defect, THRESHOLDS, trend, show_trend, direction_block, verdict_vs_first,
                   LOWER_IS_WORSE, SAME)


class Necking(Defect):
    """넥(마스크 입구)이 좁아지는 현상.

    입구가 좁아지면 이온이 들어가지 못해 식각이 멈추고(etch stop), 폴리머가 쌓여
    막히면(clogging) 아래쪽이 아예 뚫리지 않는다. 그래서 **넥 CD 는 클수록 좋다** —
    bowing·taper 와 방향이 반대다.

    두 가지 입력을 받는다:
      (neck_cd, top_cd)  — 넥/상단 비로 판정 (절대 CD 가 있을 때)
      (neck_cd,)         — 넥 CD 자체로 조건 간 비교만 (상대값만 있을 때)
    """
    name = "necking"
    label = "네킹(입구 좁아짐)"
    requires = [("neck_cd", "top_cd"), ("neck_cd",)]
    definition = "마스크 입구(넥)가 좁아져 이온 유입이 막히는 현상. 심하면 식각 중단·미개구로 이어진다."
    metric = "neck_cd"
    metric_definition = "넥 CD (작을수록 심함). top_cd 가 함께 있으면 넥CD/상단CD 비도 계산한다"
    direction = LOWER_IS_WORSE
    direction_note = "값이 클수록 입구가 열려 있음"

    def rule(self, series, judgeable):
        rows, ref = [], None
        for c in series["conditions"]:
            j = judgeable.get(c["id"], {})
            if not j.get("ok"):
                rows.append({"condition": c["id"], "level": c.get("level"),
                             "status": "판정 불가", "reason": j.get("reason")})
                continue
            n = c["neck_cd"]
            row = {"condition": c["id"], "level": c.get("level"), "neck_cd": n}
            top = c.get("top_cd")
            if top:
                r = n / top
                row["neck_ratio"] = round(r, 3)
                row["necking"] = r < THRESHOLDS["neck_ratio"]
                row["neck_ratio_threshold"] = THRESHOLDS["neck_ratio"]
            if ref is None:
                ref = n
            row["change_vs_first_pct"] = round((n / ref - 1) * 100, 1)
            row["change_vs_first"] = verdict_vs_first(n, ref, self.direction,
                                                      tol=THRESHOLDS["cd_change"])
            row["significant_vs_first"] = row["change_vs_first"] != SAME
            rows.append(row)
        out = {"defect": self.name, **self.meta(),
               "threshold_pct": THRESHOLDS["cd_change"] * 100,
               "change_vs_first_basis": "첫 조건 대비 넥 CD 비율 변화. 3% 이내는 '변화 없음'",
               "rows": rows}
        out.update(direction_block([(r["condition"], r.get(self.metric)) for r in rows],
                                   self.direction, self.direction_note))
        if show_trend(series):      # 한 변수 스윕일 때만 추세가 의미 있음
            out["trend_neck_cd"] = trend([r.get("neck_cd") for r in rows])
        return out
