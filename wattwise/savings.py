"""Turn detections into money: avoidable energy, root-cause attribution and recommendations.

Avoidable energy is split into three transparent buckets so nothing is double counted:
  1. Anomalous excess   — energy above the expected level during hours flagged as waste,
                          reported as a range: conservative (above the P90 band) to expected (above P50).
  2. Standby / always-on — how much the nightly floor exceeds the household's own best nights in
                          the same season (rolling 28-day 10th percentile), outside flagged hours.
  3. Peak shifting       — with a time-of-use tariff, the cost saved by moving flexible loads
                          (laundry, water heating, dishwasher) to off-peak hours (energy unchanged).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .anomaly import LABEL_STANDBY, LABEL_WASTE, WASTE_LABELS
from .config import CIRCUIT_LABELS, EconomicSettings

FLEX_SHARE = {"c_laundry": 1.0, "c_water_heater_ac": 0.5, "c_kitchen": 0.3}  # shiftable share per circuit


def _hour_rates(index: pd.DatetimeIndex, econ: EconomicSettings) -> np.ndarray:
    return np.array([econ.rate_for_hour(h) for h in index.hour], dtype=float)


def _circuit_expected(f: pd.DataFrame, detector) -> pd.DataFrame:
    """Context-typical kWh for every circuit (inverse of the log-ratio statistics the detector learned)."""
    out = pd.DataFrame(index=f.index)
    stats = getattr(detector.residual, "circ_stats_", {})
    for c, (med, _) in stats.items():
        if c in f.columns:
            out[c] = ((np.exp(f["context"].map(med)) - 0.02).clip(lower=0) * f["scale"]).to_numpy()
    if "c_other" in f.columns and len(out.columns):
        # 'other' = total expectation minus the metered circuits
        out["c_other"] = (f["p50"] - out.sum(axis=1)).clip(lower=0)
    return out


def _cause(row, has_circuits: bool) -> tuple[str, str]:
    """(short cause, actionable advice) for an event."""
    night = row["start"].hour <= 5 or row["start"].hour >= 23
    weekday_day = row["start"].dayofweek < 5 and 9 <= row["start"].hour <= 16
    if row["kind"] == "standby":
        return ("Always-on load rose (new standby device or appliance fault)",
                "Walk round at night: look for chargers, set-top boxes, a second fridge or a heater left on; check fridge door seals.")
    top = row.get("top_circuit")
    if has_circuits and isinstance(top, str):
        if top == "c_water_heater_ac":
            if night:
                return ("Water heater / AC running overnight",
                        "Put the geyser/AC on a timer or lower the thermostat; avoid heating water through the night.")
            if weekday_day:
                return ("Water heater / AC running while the home is likely empty",
                        "Use a timer or smart plug so cooling/heating switches off when nobody is home.")
            return ("Water heater / AC used much longer than usual", "Check thermostat set-point and running time.")
        if top == "c_kitchen":
            return ("Kitchen appliance (oven, dishwasher, microwave) at unusual time or length",
                    "Check the oven/dishwasher was not left on; run full loads on eco mode.")
        if top == "c_laundry":
            return ("Laundry appliance or fridge drawing more than usual",
                    "Avoid the tumble dryer where possible; check fridge temperature and door seals.")
        if top == "c_other":
            return ("Unmetered loads (lighting, electronics, portable heaters)",
                    "Switch off lights/electronics in empty rooms; look for plug-in heaters or pumps.")
        name = top[2:].replace("_", " ")
        when = "overnight" if night else "for longer than usual"
        return (f"'{name}' circuit running {when}", f"Check what is connected to the '{name}' circuit and whether it needs a timer.")
    if night and row["hours"] >= 3:
        return ("Device left running overnight", "Something large stayed on at night: check heaters, AC, pumps, geyser.")
    if row["hours"] <= 2:
        return ("Short burst of a high-power appliance", "Normal if intentional (kettle, iron, oven); otherwise check for faults.")
    if weekday_day:
        return ("Consumption while the home is likely empty", "Use timers/smart plugs so big loads stop when you leave.")
    return ("Unusually long period of high consumption", "Compare with what was running at that time; consider timers.")


def group_events(frame: pd.DataFrame, detector, econ: EconomicSettings) -> pd.DataFrame:
    """Merge flagged hours into events with size, cost, attribution and a plain-language cause."""
    has_circ = any(c.startswith("c_") and c != "c_other" for c in frame.columns)
    exp_c = _circuit_expected(frame, detector) if has_circ else pd.DataFrame(index=frame.index)
    rates = pd.Series(_hour_rates(frame.index, econ), index=frame.index)
    events = []
    for kind, label in (("waste", LABEL_WASTE), ("standby", LABEL_STANDBY)):
        flag = frame["label"].eq(label)
        if not flag.any():
            continue
        # merge runs separated by a single unflagged hour
        bridged = flag | (flag.shift(1, fill_value=False) & flag.shift(-1, fill_value=False))
        run_id = (bridged != bridged.shift()).cumsum()
        for _, idx in run_id[bridged].groupby(run_id[bridged]):
            g = frame.loc[idx.index]
            g = g[g["label"].eq(label) | g["kwh"].notna()]
            if kind == "waste":
                exc_hi = (g["kwh"] - g["p50"]).clip(lower=0)
                exc_lo = (g["kwh"] - g["p90"]).clip(lower=0)
            else:
                delta = (g["baseload_kw"] - g["baseload_ref_kw"]).clip(lower=0)
                exc_hi = delta
                exc_lo = (delta - 0.05).clip(lower=0)
            ev = {
                "kind": kind, "start": g.index[0], "end": g.index[-1], "hours": len(g),
                "actual_kwh": float(g["kwh"].sum()), "expected_kwh": float(g["p50"].sum()),
                "excess_kwh_low": float(exc_lo.sum()), "excess_kwh_high": float(exc_hi.sum()),
                "cost_high": float((exc_hi * rates.loc[g.index]).sum()),
                "peak_z": float(g["z"].max()) if "z" in g else np.nan,
            }
            ev["co2_kg_high"] = ev["excess_kwh_high"] * econ.co2_kg_per_kwh
            if has_circ and kind == "waste":
                exc = {}
                for c in exp_c.columns:
                    if c in g.columns:
                        exc[c] = float((g[c] - exp_c.loc[g.index, c]).clip(lower=0).sum())
                tot = sum(exc.values())
                if tot > 0:
                    shares = {c: v / tot for c, v in exc.items()}
                    ev["top_circuit"] = max(shares, key=shares.get)
                    ev["attribution"] = {CIRCUIT_LABELS.get(c[2:], c[2:]): round(v, 3) for c, v in sorted(shares.items(), key=lambda kv: -kv[1])}
            events.append(ev)
    if not events:
        return pd.DataFrame(columns=["kind", "start", "end", "hours", "actual_kwh", "expected_kwh", "excess_kwh_low",
                                     "excess_kwh_high", "cost_high", "peak_z", "co2_kg_high", "cause", "advice", "severity"])
    ev = pd.DataFrame(events).sort_values("start").reset_index(drop=True)
    causes = ev.apply(lambda r: _cause(r, has_circ), axis=1)
    ev["cause"] = [c[0] for c in causes]
    ev["advice"] = [c[1] for c in causes]
    ev["severity"] = pd.cut(ev["excess_kwh_high"], [-1, 1.0, 3.0, np.inf], labels=["Low", "Medium", "High"]).astype(str)
    return ev


def standby_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily always-on floor vs the household's own best nights in the same season."""
    daily = frame["kwh"].groupby(frame.index.normalize())
    base = daily.apply(lambda v: np.nan if v.count() < 18 else np.sort(v.dropna().to_numpy())[:3].mean())
    base.index = pd.DatetimeIndex(base.index)
    best = base.rolling(28, min_periods=7, center=True).quantile(0.10)
    best = best.fillna(base.quantile(0.10))
    waste_hours = frame["label"].isin(WASTE_LABELS).groupby(frame.index.normalize()).sum()
    waste_hours.index = pd.DatetimeIndex(waste_hours.index)
    obs_hours = daily.count()
    obs_hours.index = pd.DatetimeIndex(obs_hours.index)
    excess_kw = (base - best).clip(lower=0)
    # standby flagged as an event is counted in the event bucket, not here
    standby_days = frame["label"].eq(LABEL_STANDBY).groupby(frame.index.normalize()).any()
    standby_days.index = pd.DatetimeIndex(standby_days.index)
    usable_h = (obs_hours - waste_hours).clip(lower=0).where(~standby_days, 0)
    return pd.DataFrame({
        "baseload_kw": base, "best_nights_kw": best, "excess_kw": excess_kw,
        "avoidable_kwh": excess_kw * usable_h, "daily_kwh": daily.sum(min_count=18),
    })


def peak_shift(frame: pd.DataFrame, econ: EconomicSettings, flex_share_no_circuits: float = 0.15) -> dict:
    if not econ.use_tou:
        return {"shiftable_kwh": 0.0, "cost_saving": 0.0, "basis": "flat tariff — no time-of-use saving"}
    peak = frame.index.hour.isin(econ.peak_hours) & frame["kwh"].notna().to_numpy()
    spread = econ.peak_rate - econ.offpeak_rate
    circ = [c for c in FLEX_SHARE if c in frame.columns]
    if circ:
        kwh = float(sum(frame.loc[peak, c].sum() * FLEX_SHARE[c] for c in circ))
        basis = "flexible circuits used at peak (laundry 100%, water heating 50%, dishwasher 30%)"
    else:
        base = frame["kwh"].rolling(24 * 7, min_periods=24).quantile(0.05)
        kwh = float(((frame["kwh"] - base).clip(lower=0)[peak]).sum() * flex_share_no_circuits)
        basis = f"{flex_share_no_circuits:.0%} of above-baseload peak-hour usage assumed movable"
    return {"shiftable_kwh": kwh, "cost_saving": kwh * spread, "basis": basis}


def grade(avoidable_share: float) -> tuple[int, str]:
    score = int(round(np.clip(100 - avoidable_share * 400, 0, 100)))  # 5 % avoidable -> 80, 15 % -> 40
    letter = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D" if score >= 40 else "E"
    return score, letter


def summarize(frame: pd.DataFrame, events: pd.DataFrame, standby: pd.DataFrame, econ: EconomicSettings,
              flex_share_no_circuits: float = 0.15) -> dict:
    total = float(frame["kwh"].sum())
    days = max(frame["kwh"].notna().sum() / 24.0, 1e-9)
    annual = 365.0 / days
    rates = _hour_rates(frame.index, econ)
    total_cost = float(np.nansum(frame["kwh"].to_numpy() * rates))
    avg_rate = total_cost / total if total else econ.flat_rate

    ev_lo = float(events["excess_kwh_low"].sum()) if len(events) else 0.0
    ev_hi = float(events["excess_kwh_high"].sum()) if len(events) else 0.0
    ev_cost = float(events["cost_high"].sum()) if len(events) else 0.0
    sb = float(standby["avoidable_kwh"].sum())
    ps = peak_shift(frame, econ, flex_share_no_circuits)
    av_lo, av_hi = ev_lo + sb, ev_hi + sb
    share = av_hi / total if total else 0.0
    score, letter = grade(share)
    s = {
        "days_analyzed": round(days, 1), "total_kwh": total, "total_cost": total_cost, "avg_rate": avg_rate,
        "avg_daily_kwh": total / days,
        "median_baseload_w": float(standby["baseload_kw"].median() * 1000),
        "best_baseload_w": float(standby["best_nights_kw"].median() * 1000),
        "events": int((events["kind"] == "waste").sum()) if len(events) else 0,
        "standby_events": int((events["kind"] == "standby").sum()) if len(events) else 0,
        "waste_hours": int(frame["label"].eq(LABEL_WASTE).sum()),
        "anomaly_kwh_low": ev_lo, "anomaly_kwh_high": ev_hi, "standby_kwh": sb,
        "avoidable_kwh_low": av_lo, "avoidable_kwh_high": av_hi,
        "avoidable_cost": ev_cost + sb * avg_rate, "peak_shift_cost": ps["cost_saving"], "peak_shift_kwh": ps["shiftable_kwh"],
        "peak_shift_basis": ps["basis"],
        "avoidable_share": share, "co2_kg": av_hi * econ.co2_kg_per_kwh,
        "annual_factor": annual, "efficiency_score": score, "efficiency_grade": letter,
    }
    s["annual_avoidable_kwh"] = av_hi * annual
    s["annual_saving"] = (s["avoidable_cost"] + ps["cost_saving"]) * annual
    s["annual_co2_kg"] = s["co2_kg"] * annual
    return s


def recommendations(events: pd.DataFrame, standby: pd.DataFrame, summary: dict, econ: EconomicSettings) -> list[dict]:
    recs = []
    k = summary["annual_factor"]
    if len(events):
        for cause, g in events.groupby("cause"):
            kwh = float(g["excess_kwh_high"].sum()) * k
            recs.append({"title": cause, "action": g["advice"].iloc[0], "occurrences": int(len(g)),
                         "annual_kwh": kwh, "annual_saving": float(g["cost_high"].sum()) * k})
    if summary["standby_kwh"] > 0:
        gap_w = max(summary["median_baseload_w"] - summary["best_baseload_w"], 0)
        recs.append({
            "title": f"Always-on load: typical {summary['median_baseload_w']:.0f} W vs {summary['best_baseload_w']:.0f} W on your best nights",
            "action": f"Cutting ~{gap_w:.0f} W of standby (smart power strips, unplugging idle chargers/set-top boxes, efficient fridge) "
                      "brings every night down to your best-night level.",
            "occurrences": int((standby["excess_kw"] > 0.02).sum()),
            "annual_kwh": summary["standby_kwh"] * k, "annual_saving": summary["standby_kwh"] * summary["avg_rate"] * k})
    if summary["peak_shift_cost"] > 0:
        recs.append({"title": "Shift flexible loads to off-peak hours",
                     "action": f"Run washer/dryer, dishwasher and water heating outside {min(econ.peak_hours):02d}:00–{max(econ.peak_hours) + 1:02d}:00 "
                               f"({summary['peak_shift_basis']}).",
                     "occurrences": None, "annual_kwh": 0.0, "annual_saving": summary["peak_shift_cost"] * k})
    return sorted(recs, key=lambda r: -r["annual_saving"])


def standby_whatif(watts: float, econ: EconomicSettings) -> dict:
    kwh = watts / 1000 * 8760
    return {"annual_kwh": kwh, "annual_saving": kwh * econ.flat_rate, "annual_co2_kg": kwh * econ.co2_kg_per_kwh}
