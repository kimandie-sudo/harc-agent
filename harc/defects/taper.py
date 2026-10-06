from .base import (Defect, THRESHOLDS, trend, show_trend, direction_block, verdict_vs_first,
                   HIGHER_IS_WORSE, SAME)

class Taper(Defect):
    name = "taper"          # v1 의 taper_narrowing 을 이 이름으로 통일
    label = "테이퍼(하단 좁아짐)"
    requires = [("top_cd", "bottom_cd"), ("top_bottom_ratio",)]
    definition = "아래로 갈수록 폭이 좁아지는 현상."
    metric = "top_bottom_ratio"
    metric_definition = "상단CD/하단CD (클수록 심함)"
    direction = HIGHER_IS_WORSE
    direction_note = "값이 작을수록 수직"

    def rule(self, series, judgeable):
        rows, ref = [], None
        for c in series["conditions"]:
            j = judgeable.get(c["id"], {})
            if not j.get("ok"):
                rows.append({"condition": c["id"], "level": c.get("level"), "status": "판정 불가",
                             "reason": j.get("reason")})
                continue
            r = c.get("top_bottom_ratio") or c["top_cd"] / c["bottom_cd"]
            row = {"condition": c["id"], "level": c.get("level"), "top_bottom_ratio": round(r, 2)}
            if c.get("bottom_cd") is not None:
                row.update(top_cd=c["top_cd"], bottom_cd=c["bottom_cd"])
            if ref is None:
                ref = r
            row["change_vs_first_pct"] = round((r / ref - 1) * 100, 1)
            row["change_vs_first"] = verdict_vs_first(r, ref, self.direction,
                                                      tol=THRESHOLDS["taper_change"])
            row["significant_vs_first"] = row["change_vs_first"] != SAME
            rows.append(row)
        out = {"defect": self.name, **self.meta(), "threshold_pct": THRESHOLDS["taper_change"] * 100,
               "change_vs_first_basis": "첫 조건 대비 비율 변화. 8% 이내는 '변화 없음'",
               "rows": rows}
        out.update(direction_block([(r["condition"], r.get(self.metric)) for r in rows],
                                   self.direction, self.direction_note))
        f = series.get("factors")
        if show_trend(series):  # 한 변수 스윕일 때만 추세가 의미 있음
            out["trend_ratio"] = trend([r.get("top_bottom_ratio") for r in rows])
        # 2x2 요인 설계면 주효과·상호작용(비율 기준) 계산
        if f and len(f) == 2:
            out["factorial"] = factorial_effects(series, {r["condition"]: r.get("top_bottom_ratio")
                                                          for r in rows}, self.direction)
        return out

def factorial_effects(series, val, direction=HIGHER_IS_WORSE):
    (fa, ma), (fb, mb) = list(series["factors"].items())
    la, lb = list(dict.fromkeys(ma.values())), list(dict.fromkeys(mb.values()))  # 첫 조건 수준이 기준
    if len(la) != 2 or len(lb) != 2:
        return None
    cell = {}
    for cid, v in val.items():
        if v is not None:
            cell[(ma[cid], mb[cid])] = v
    if len(cell) != 4:
        return None
    eff = {}
    for fname, levels, idx in ((fa, la, 0), (fb, lb, 1)):
        other = lb if idx == 0 else la
        ch = []
        for o in other:
            k0 = (levels[0], o) if idx == 0 else (o, levels[0])
            k1 = (levels[1], o) if idx == 0 else (o, levels[1])
            ch.append({"at": o, "from": levels[0], "to": levels[1],
                       "change_pct": round((cell[k1] / cell[k0] - 1) * 100, 1),
                       "verdict": verdict_vs_first(cell[k1], cell[k0], direction,
                                                   tol=THRESHOLDS["taper_change"])})
        eff[fname] = ch
    return {"main_effects_pct": eff, "direction": direction,
            "note": "verdict 는 위 direction 기준. 같은 요인의 두 변화율이 크게 다르면 상호작용 가능성. "
                    "비율 지표 기준이며 보잉 등 다른 지표에선 결론이 다를 수 있음"}
