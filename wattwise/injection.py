"""Synthetic waste injection — a labelled benchmark for unsupervised detectors.

Real meter data has no ground-truth anomaly labels, so we plant realistic waste events with
known timing and size into held-out data, re-run the full pipeline, and measure how many the
detectors recover and how often they raise false alarms on genuine (normal) peaks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WASTE_TYPES = {
    "appliance_left_on_overnight": "Heater / AC / pump left running overnight (+0.8–1.5 kW for 4–10 h)",
    "high_power_spike": "Short burst of a high-power appliance (+1.5–3 kW for 1–2 h)",
    "daytime_while_away": "Cooling/heating running in an empty home on a weekday (+0.7–1.2 kW for 6–9 h)",
    "standby_creep": "New always-on load, e.g. faulty fridge or forgotten device (+0.25–0.45 kW for 2–4 days)",
}
TYPE_CIRCUIT = {
    "appliance_left_on_overnight": "c_water_heater_ac",
    "high_power_spike": "c_kitchen",
    "daytime_while_away": "c_water_heater_ac",
    "standby_creep": "c_other",
}


def inject(hourly: pd.DataFrame, n_per_type: int = 6, seed: int = 0, start=None, end=None):
    """Return (modified copy, per-hour label Series of type names or '', events DataFrame)."""
    rng = np.random.default_rng(seed)
    df = hourly.copy()
    lab = pd.Series("", index=df.index, dtype=object)
    win = df.loc[start:end]
    valid_days = win["kwh"].resample("D").count()
    days = list(valid_days[valid_days == 24].index)
    events = []
    occupied = set()

    def free(idx):
        return not any(t in occupied for t in idx) and df.loc[idx, "kwh"].notna().all()

    for etype in WASTE_TYPES:
        placed, tries = 0, 0
        while placed < n_per_type and tries < 500:
            tries += 1
            d = days[rng.integers(len(days))]
            if etype == "appliance_left_on_overnight":
                h0, dur, amt = int(rng.integers(0, 3)), int(rng.integers(4, 11)), rng.uniform(0.8, 1.5)
            elif etype == "high_power_spike":
                h0, dur, amt = int(rng.integers(0, 24)), int(rng.integers(1, 3)), rng.uniform(1.5, 3.0)
            elif etype == "daytime_while_away":
                if d.dayofweek >= 5:
                    continue
                h0, dur, amt = int(rng.integers(9, 12)), int(rng.integers(6, 10)), rng.uniform(0.7, 1.2)
            else:
                h0, dur, amt = int(rng.integers(0, 24)), int(rng.integers(48, 97)), rng.uniform(0.25, 0.45)
            idx = pd.date_range(d + pd.Timedelta(hours=h0), periods=dur, freq="h")
            if idx[-1] not in df.index or not free(idx):
                continue
            df.loc[idx, "kwh"] += amt
            circ = TYPE_CIRCUIT[etype]
            if circ in df.columns:
                df.loc[idx, circ] += amt
            lab.loc[idx] = etype
            occupied.update(idx)
            events.append({"type": etype, "start": idx[0], "end": idx[-1], "hours": dur,
                           "kw_added": round(float(amt), 3), "kwh_added": round(float(amt * dur), 2)})
            placed += 1
    return df, lab, pd.DataFrame(events)


def evaluate_flags(flags: pd.Series, labels: pd.Series, f: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Hour-level and event-level scores of a boolean flag series against injected labels."""
    valid = f["kwh"].notna()
    flags = flags.fillna(False).astype(bool) & valid
    pos = (labels != "") & valid
    tp = int((flags & pos).sum())
    fp = int((flags & ~pos).sum())
    fn = int((~flags & pos).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    per_type = {}
    for t, g in events.groupby("type"):
        hit = [flags.loc[r.start:r.end].any() for r in g.itertuples()]
        per_type[t] = float(np.mean(hit))
    ev_recall = float(np.mean([flags.loc[r.start:r.end].any() for r in events.itertuples()]))

    # false alarms: flagged runs that do not touch an injected event
    runs = (flags != flags.shift()).cumsum()[flags]
    false_runs = sum(1 for _, idx in runs.groupby(runs) if not pos.loc[idx.index].any())
    weeks = valid.sum() / (24 * 7)

    # how often the household's ordinary peak hours (top 5 % of all un-injected hours) raise an alert:
    # a naive "high usage" alarm fires on all of them; a context-aware detector should not.
    clean = valid & ~pos
    peak = clean & (f["kwh"] > f.loc[clean, "kwh"].quantile(0.95))
    peak_far = float(flags[peak].mean()) if peak.any() else 0.0

    return {
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        "event_recall": round(ev_recall, 3),
        "false_alarm_events_per_week": round(false_runs / weeks, 2),
        "alert_rate_on_normal_peak_hours_%": round(peak_far * 100, 1),
        "event_recall_by_type": {k: round(v, 3) for k, v in per_type.items()},
    }
