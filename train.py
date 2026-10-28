"""Train, tune and honestly evaluate every WattWise model, then save artifacts and reports.

    python train.py                       # uses data/processed/hourly.parquet (or builds it from the raw file)
    python train.py --raw path/to/household_power_consumption.txt

Protocol (strictly chronological):
    train        2006-12 … 2009-12   fit forecaster + anomaly detectors
    calibration  2010-01 … 2010-06   tune LightGBM, conformal margins, alert thresholds
    test         2010-07 … 2010-11   report every number in reports/ (never used for any choice)
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wattwise.anomaly import (WASTE_LABELS, EnsembleDetector, Thresholds, build_frame,
                              hour_of_day_threshold, naive_global_threshold)
from wattwise.config import (CAL_END, LGB_PARAMS, MODELS_DIR, PROCESSED_FILE, RAW_FILE, REPORTS_DIR,
                             SAMPLE_DIR, TRAIN_END)
from wattwise.features import FEATURES, build_features, target
from wattwise.forecast import Forecaster, interval_metrics, naive_baselines, point_metrics
from wattwise.injection import WASTE_TYPES, evaluate_flags, inject
from wattwise.io import load_uci, to_generic_csv

warnings.filterwarnings("ignore")
CAL_START = "2010-01-01"
TEST_START = "2010-07-01"
FINAL_TRAIN_END = "2010-08-31 23:00"
FA_BUDGET = 1.0  # max false-alarm events per week allowed when tuning any detector
COLORS = {"actual": "#1f2933", "fc": "#2563eb", "band": "#93c5fd", "waste": "#dc2626", "ok": "#16a34a", "grey": "#9ca3af"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================== forecasting
def tune_lightgbm(y):
    """Small time-series hyper-parameter search: train on train, score day-ahead MAE on calibration."""
    feats = build_features(y, "FR")
    tgt = target(y, feats)
    ok = tgt.notna() & feats["scale"].notna() & feats["lag24"].notna()
    tr = ok & (feats.index <= TRAIN_END)
    cal = ok & (feats.index > TRAIN_END) & (feats.index <= CAL_END)
    from lightgbm import LGBMRegressor
    results = []
    for leaves, n_est, mcs in itertools.product([15, 31, 63], [300, 600, 1000], [20, 60]):
        p = dict(LGB_PARAMS, num_leaves=leaves, n_estimators=n_est, min_child_samples=mcs)
        m = LGBMRegressor(objective="quantile", alpha=0.5, **p).fit(feats.loc[tr, FEATURES], tgt[tr])
        pred = m.predict(feats.loc[cal, FEATURES]) * feats.loc[cal, "scale"]
        mae = float(np.mean(np.abs(pred - y[cal])))
        results.append({"num_leaves": leaves, "n_estimators": n_est, "min_child_samples": mcs, "cal_MAE": mae})
    res = pd.DataFrame(results).sort_values("cal_MAE")
    best = res.iloc[0]
    params = dict(LGB_PARAMS, num_leaves=int(best.num_leaves), n_estimators=int(best.n_estimators),
                  min_child_samples=int(best.min_child_samples))
    return params, res


def ridge_dayahead(y, test_idx):
    feats = build_features(y, "FR")
    tgt = target(y, feats)
    ok = tgt.notna() & feats["scale"].notna() & (feats.index <= TRAIN_END)
    X = pd.get_dummies(feats[FEATURES], columns=["hour", "dow"])
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(X[ok], tgt[ok])
    Xt = X.reindex(test_idx)
    pred = model.predict(Xt) * feats["scale"].reindex(test_idx).to_numpy()
    return pd.Series(np.clip(pred, 0, None), index=test_idx)


def evaluate_forecasting(y, params):
    log("Forecaster: fitting on train period")
    fc = Forecaster("FR", params=params).fit(y, end=TRAIN_END)
    log("Forecaster: conformal calibration (7-day recursive backtests over Jan–Jun 2010)")
    fc.calibrate(y, CAL_START, pd.Timestamp(CAL_END) - pd.Timedelta(days=7), horizon=7)

    # ---- day-ahead comparison on test
    ins = fc.predict_insample(y)
    test_idx = y.loc[TEST_START:].dropna().index
    test_idx = test_idx[ins.loc[test_idx, "p50"].notna()]
    act = y.reindex(test_idx)
    table = {"WattWise LightGBM (quantile P50)": point_metrics(act, ins.loc[test_idx, "p50"])}
    table["Ridge regression (same features)"] = point_metrics(act, ridge_dayahead(y, test_idx))
    nb = naive_baselines(y, test_idx)
    for c in nb:
        table[c] = point_metrics(act, nb[c])
    da_int = interval_metrics(act, ins.loc[test_idx, "p10"], ins.loc[test_idx, "p90"], ins.loc[test_idx, "p50"])

    # daily-total accuracy (what a monthly bill projection depends on)
    dsum = pd.DataFrame({"a": act, "p": ins.loc[test_idx, "p50"], "n": nb["Seasonal naive (same hour last week)"]})
    dsum = dsum.groupby(dsum.index.normalize()).filter(lambda g: len(g) == 24)
    daily = dsum.groupby(dsum.index.normalize()).sum()
    daily_metrics = {"WattWise": point_metrics(daily.a, daily.p), "Seasonal naive": point_metrics(daily.a, daily.n)}

    # ---- multi-day recursive forecast on test, per lead day
    log("Forecaster: 7-day recursive backtest on test period")
    bt = fc.backtest(y, TEST_START, y.index[-1] - pd.Timedelta(days=7), horizon=7)
    yy = y.asfreq("h")
    bt["naive_week"] = yy.shift(168).reindex(bt.index).to_numpy()
    per_lead = []
    for lead, g in bt.groupby("lead"):
        m = point_metrics(g.actual, g.p50)
        im = interval_metrics(g.actual, g.p10, g.p90, g.p50)
        nm = point_metrics(g.actual, g.naive_week)
        per_lead.append({"lead_day": int(lead), "MAE": m["MAE"], "WAPE_%": m["WAPE_%"],
                         "naive_week_MAE": nm["MAE"], "coverage_80_%": im["coverage_80_%"],
                         "width_kWh": im["mean_width_kWh"], "margin_norm": fc.margins.get(int(lead))})
    per_lead = pd.DataFrame(per_lead)
    dd = bt.assign(day=bt.index.normalize()).groupby(["origin", "lead", "day"]).agg(a=("actual", "sum"), n=("actual", "count"))
    dd = dd[dd.n == 24].reset_index()
    cov_daily = []
    for (o, l), g in dd.groupby(["origin", "lead"]):
        fcd = fc.daily_totals(bt[(bt.origin == o) & (bt.lead == l)])
        if len(fcd):
            cov_daily.append({"lead": l, "hit": float((g.a.iloc[0] >= fcd.p10.iloc[0]) & (g.a.iloc[0] <= fcd.p90.iloc[0]))})
    cd = pd.DataFrame(cov_daily).groupby("lead").hit.mean() * 100
    per_lead["daily_total_coverage_80_%"] = per_lead.lead_day.map(cd)

    # ---- rolling (online) conformal: margins re-estimated from the last 42 days of known errors
    log("Forecaster: rolling-conformal evaluation on test period")
    cal_bt = fc.backtest(y, CAL_START, pd.Timestamp(CAL_END) - pd.Timedelta(days=7), horizon=7, calibrated=False)
    raw = fc.backtest(y, TEST_START, y.index[-1] - pd.Timedelta(days=7), horizon=7, calibrated=False)
    hist = pd.concat([cal_bt, raw])
    hist["score"] = Forecaster.conformity(hist)
    roll_cov = {}
    for lead in range(1, 8):
        a, t = hist[hist.lead == lead], raw[raw.lead == lead]
        hits, widths = [], []
        for o, g in t.groupby("origin"):
            known = a[(a.index < o) & (a.index >= o - pd.Timedelta(days=42))]
            m = fc._margin(known["score"])
            lo, hi = (g.p10 - m * g.scale).clip(lower=0), g.p90 + m * g.scale
            hits.append(((g.actual >= lo) & (g.actual <= hi)).to_numpy())
            widths.append((hi - lo).to_numpy())
        roll_cov[lead] = (float(np.concatenate(hits).mean() * 100), float(np.concatenate(widths).mean()))
    per_lead["rolling_coverage_80_%"] = per_lead.lead_day.map(lambda l: roll_cov[l][0])
    per_lead["rolling_width_kWh"] = per_lead.lead_day.map(lambda l: roll_cov[l][1])
    return fc, ins, table, da_int, daily_metrics, per_lead, bt


# =============================================================================== anomaly detection
def detector_flag_sets(ens, s, fi, ref):
    """Candidate flag series for every detector at a range of thresholds (for fair tuning)."""
    cands = {}
    for pct in (97, 98, 99, 99.5, 99.8):
        cands[("Naive: global high-usage threshold", pct)] = naive_global_threshold(fi, ref, pct)
        cands[("Naive: hour-of-day threshold", pct)] = hour_of_day_threshold(fi, ref, pct)
    for p in (0.95, 0.97, 0.98, 0.99, 0.995, 0.999):
        cands[("Isolation Forest only", p)] = s["p_if"] >= p
        cands[("Autoencoder only", p)] = s["p_ae"] >= p
    for z in (2.5, 3, 3.5, 4, 5, 6, 8):
        cands[("Contextual residual only", z)] = (s["z"] >= z) & (fi["kwh"] > fi["p90"])
    return cands


def ensemble_grid():
    for z, p, zsus, zb in itertools.product([2.5, 3, 3.5, 4], [0.97, 0.98, 0.99], [3, 3.5, 4, 5], [3, 4, 5]):
        yield Thresholds(z=z, p=p, z_strong=99.0, z_sustained=zsus, z_base=zb)


def aggregate(results):
    keys = ["precision", "recall", "f1", "event_recall", "false_alarm_events_per_week", "alert_rate_on_normal_peak_hours_%"]
    out = {k: float(np.mean([r[k] for r in results])) for k in keys}
    types = results[0]["event_recall_by_type"].keys()
    out["event_recall_by_type"] = {t: float(np.mean([r["event_recall_by_type"][t] for r in results])) for t in types}
    return out


def pick_best(rows):
    """Highest event recall + F1 within the false-alarm budget (fallback: fewest false alarms)."""
    ok = [r for r in rows if r[1]["false_alarm_events_per_week"] <= FA_BUDGET]
    pool = ok or sorted(rows, key=lambda r: r[1]["false_alarm_events_per_week"])[:5]
    return max(pool, key=lambda r: r[1]["event_recall"] + r[1]["f1"])


def injected_sets(hourly, fc, ens, start, end, seeds, n_per_type):
    sets = []
    for seed in seeds:
        hi, lab, ev = inject(hourly, n_per_type=n_per_type, seed=seed, start=start, end=end)
        fi = build_frame(hi, fc.predict_insample(hi["kwh"])).loc[start:end]
        sets.append((fi, ens.raw_scores(fi), lab.loc[fi.index], ev))
    return sets


def evaluate_anomaly(hourly, fc):
    log("Anomaly: fitting detectors on train period")
    f_all = build_frame(hourly, fc.predict_insample(hourly["kwh"]))
    ref = f_all.loc[:TRAIN_END]
    ens = EnsembleDetector().fit(ref)

    log("Anomaly: tuning thresholds on calibration period with injected waste (3 seeds)")
    cal_sets = injected_sets(hourly, fc, ens, CAL_START, CAL_END, seeds=(11, 12, 13), n_per_type=8)
    test_sets = injected_sets(hourly, fc, ens, TEST_START, None, seeds=(101, 102, 103, 104, 105), n_per_type=8)

    # ---- ensemble
    rows = []
    for th in ensemble_grid():
        res = aggregate([evaluate_flags(ens.label(fi, s, th).label.isin(WASTE_LABELS), l, fi, ev) for fi, s, l, ev in cal_sets])
        rows.append((th, res))
    best_th, best_cal = pick_best(rows)
    ens.th = best_th
    log(f"Anomaly: chosen thresholds {best_th}")

    # ---- each single detector / naive baseline, tuned the same way
    chosen = {"WattWise ensemble": best_th}
    cal_cands = [detector_flag_sets(ens, s, fi, ref) for fi, s, l, ev in cal_sets]
    names = sorted({k[0] for k in cal_cands[0]})
    for name in names:
        rws = []
        for key in [k for k in cal_cands[0] if k[0] == name]:
            res = aggregate([evaluate_flags(c[key], l, fi, ev) for c, (fi, s, l, ev) in zip(cal_cands, cal_sets)])
            rws.append((key[1], res))
        chosen[name] = pick_best(rws)[0]

    # ---- honest test evaluation
    log("Anomaly: evaluating on test period (5 injection seeds)")
    test_table = {}
    test_table["WattWise ensemble"] = aggregate(
        [evaluate_flags(ens.label(fi, s).label.isin(WASTE_LABELS), l, fi, ev) for fi, s, l, ev in test_sets])
    for name in names:
        res = []
        for fi, s, l, ev in test_sets:
            c = detector_flag_sets(ens, s, fi, ref)
            res.append(evaluate_flags(c[(name, chosen[name])], l, fi, ev))
        test_table[name] = aggregate(res)
    example = test_sets[0]
    return ens, best_th, best_cal, chosen, test_table, example, f_all


# =============================================================================== figures
def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def make_figures(y, ins, per_lead, fc, test_table, example, ens):
    REPORTS_DIR.mkdir(exist_ok=True)
    # 1. a test week with calibrated band
    wk = ins.loc["2010-10-04":"2010-10-10 23:00"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(wk.index, wk.p10, wk.p90, color=COLORS["band"], alpha=0.6, label="80% interval (conformal)")
    ax.plot(wk.index, wk.p50, color=COLORS["fc"], lw=1.6, label="Forecast (P50)")
    ax.plot(wk.index, y.reindex(wk.index), color=COLORS["actual"], lw=1.1, label="Actual")
    ax.set_ylabel("kWh per hour")
    style(ax, "Day-ahead forecast vs actual — test week, Oct 2010")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "forecast_test_week.png", dpi=130); plt.close(fig)

    # 2. per-lead accuracy + coverage
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].plot(per_lead.lead_day, per_lead.MAE, "o-", color=COLORS["fc"], label="WattWise")
    axes[0].plot(per_lead.lead_day, per_lead.naive_week_MAE, "o--", color=COLORS["grey"], label="Seasonal naive")
    axes[0].set_xlabel("days ahead"); axes[0].set_ylabel("MAE (kWh/h)"); axes[0].legend(frameon=False)
    style(axes[0], "Accuracy by forecast horizon")
    w = 0.38
    axes[1].bar(per_lead.lead_day - w / 2, per_lead["coverage_80_%"], w, color=COLORS["grey"], label="static conformal")
    axes[1].bar(per_lead.lead_day + w / 2, per_lead["rolling_coverage_80_%"], w, color=COLORS["fc"], label="rolling conformal (42 d)")
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    axes[1].axhline(80, color=COLORS["actual"], ls="--", lw=1)
    axes[1].set_ylim(50, 90); axes[1].set_xlabel("days ahead"); axes[1].set_ylabel("coverage %")
    style(axes[1], "Interval coverage (target 80%)")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "forecast_horizon.png", dpi=130); plt.close(fig)

    # 3. feature importance
    fi_ = fc.feature_importance_.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(fi_.index, fi_.values / fi_.values.sum() * 100, color=COLORS["fc"])
    ax.set_xlabel("share of gain (%)")
    style(ax, "Forecaster feature importance")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=130); plt.close(fig)

    # 4. detector comparison
    tt = pd.DataFrame(test_table).T.sort_values("f1")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    cols = [COLORS["waste"] if i == "WattWise ensemble" else COLORS["grey"] for i in tt.index]
    axes[0].barh(tt.index, tt.f1, color=cols); axes[0].set_xlabel("hour-level F1")
    style(axes[0], "Planted-waste detection (test)")
    axes[1].barh(tt.index, tt.event_recall * 100, color=cols); axes[1].set_xlabel("events found (%)")
    style(axes[1], "Share of planted events found (test)")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "detector_comparison.png", dpi=130); plt.close(fig)

    # 5. example: detection on injected test data
    fi, s, l, ev = example
    out = ens.label(fi, s)
    # pick the overnight event in Oct–Nov whose surrounding 4 days are complete and least busy with other flags
    best, best_n = None, 1e9
    for e_ in ev[(ev.type == "appliance_left_on_overnight") & (ev.start >= "2010-10-01")].itertuples():
        w_ = fi.loc[e_.start - pd.Timedelta(days=2): e_.start + pd.Timedelta(days=2)]
        if w_["kwh"].isna().any():
            continue
        n_ = int(out.loc[w_.index, "label"].isin(WASTE_LABELS).sum())
        if n_ < best_n:
            best, best_n = e_, n_
    e = best if best is not None else ev[ev.type == "appliance_left_on_overnight"].iloc[0]
    win = fi.loc[e.start - pd.Timedelta(days=2): e.start + pd.Timedelta(days=2)]
    o = out.loc[win.index]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(win.index, win.p10, win.p90, color=COLORS["band"], alpha=0.5, label="expected range")
    ax.plot(win.index, win.kwh, color=COLORS["actual"], lw=1.2, label="meter")
    flag = o.label.isin(WASTE_LABELS)
    ax.scatter(win.index[flag], win.kwh[flag], color=COLORS["waste"], s=22, zorder=3, label="flagged as waste")
    ax.axvspan(e.start, e.end + pd.Timedelta(hours=1), color=COLORS["waste"], alpha=0.08, label="planted event")
    ax.set_ylabel("kWh per hour"); ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style(ax, "Detected: appliance left on overnight (planted in test data)")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "detection_example.png", dpi=130); plt.close(fig)


# =============================================================================== samples for the demo
def make_samples(hourly):
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    recent = hourly.loc["2010-09-01":]  # after FINAL_TRAIN_END: unseen by the shipped models
    to_generic_csv(recent).to_csv(SAMPLE_DIR / "sample_meter_readings.csv", index=False)
    hi, lab, ev = inject(recent, n_per_type=2, seed=7, start="2010-10-01", end=None)
    to_generic_csv(hi).to_csv(SAMPLE_DIR / "sample_with_planted_waste.csv", index=False)
    ev["description"] = ev["type"].map(WASTE_TYPES)
    ev.to_csv(SAMPLE_DIR / "planted_waste_answer_key.csv", index=False)
    # simplest possible format: timestamp + total kWh, no circuits
    to_generic_csv(recent)[["timestamp", "kwh"]].to_csv(SAMPLE_DIR / "sample_total_only.csv", index=False)
    # raw minute-level slice in the original Kaggle/UCI format
    if RAW_FILE.exists():
        raw = pd.read_csv(RAW_FILE, sep=";", dtype=str)
        dt = pd.to_datetime(raw["Date"], format="%d/%m/%Y")
        raw[(dt >= "2010-10-01")].to_csv(SAMPLE_DIR / "sample_uci_minute_format.txt", sep=";", index=False)


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_FILE))
    ap.add_argument("--skip-tuning", action="store_true")
    args = ap.parse_args()
    MODELS_DIR.mkdir(exist_ok=True); REPORTS_DIR.mkdir(exist_ok=True)

    if PROCESSED_FILE.exists():
        hourly = pd.read_parquet(PROCESSED_FILE)
    else:
        log("Building hourly series from raw minute data")
        hourly, _ = load_uci(args.raw)
        PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_parquet(PROCESSED_FILE)
    y = hourly["kwh"]
    log(f"Hourly series: {len(y)} hours, {y.isna().sum()} missing")

    if args.skip_tuning:
        params, tune_res = dict(LGB_PARAMS), None
    else:
        log("Tuning LightGBM (18 configs, chronological validation)")
        params, tune_res = tune_lightgbm(y)
        log(f"Best: {params['num_leaves']} leaves, {params['n_estimators']} trees, min_child {params['min_child_samples']}")

    fc, ins, table, da_int, daily_metrics, per_lead, bt = evaluate_forecasting(y, params)
    ens, th, th_cal, chosen, test_table, example, f_all = evaluate_anomaly(hourly, fc)
    make_figures(y, ins, per_lead, fc, test_table, example, ens)

    # ------------------------------------------------------------ final models on all data
    log("Refitting final models on all data")
    # Trained through Aug 2010 so Sep–Nov 2010 (the demo files) stay genuinely unseen.
    final_fc = Forecaster("FR", params=params).fit(y, end=FINAL_TRAIN_END)
    final_fc.margins = dict(fc.margins)  # honest conformal margins; the app re-calibrates per meter
    final_fc.daily_q_ = dict(fc.daily_q_)
    f_final = build_frame(hourly, final_fc.predict_insample(y)).loc[:FINAL_TRAIN_END]
    det_full = EnsembleDetector(th).fit(f_final)
    det_kwh = EnsembleDetector(th).fit(f_final.drop(columns=[c for c in f_final if c.startswith("c_")]))
    joblib.dump(final_fc, MODELS_DIR / "forecaster.joblib", compress=3)
    joblib.dump(det_full, MODELS_DIR / "detector_with_circuits.joblib", compress=3)
    joblib.dump(det_kwh, MODELS_DIR / "detector_total_only.joblib", compress=3)

    metrics = {
        "data": {"hours": int(len(y)), "missing_hours": int(y.isna().sum()), "start": str(y.index[0]), "end": str(y.index[-1]),
                 "split": {"train_end": TRAIN_END, "calibration": [CAL_START, CAL_END], "test_start": TEST_START}},
        "forecast": {
            "lightgbm_params": {k: params[k] for k in ("num_leaves", "n_estimators", "min_child_samples", "learning_rate")},
            "day_ahead_test": table, "day_ahead_interval_test": da_int, "daily_total_test": daily_metrics,
            "per_lead_day_test": per_lead.to_dict(orient="records"),
            "conformal_margins_by_lead": fc.margins,
            "feature_importance_top10": (fc.feature_importance_.head(10) / fc.feature_importance_.sum()).round(4).to_dict(),
        },
        "anomaly": {
            "thresholds": th.as_dict(), "calibration_score": th_cal, "baseline_settings": {k: (v.as_dict() if hasattr(v, "as_dict") else v) for k, v in chosen.items()},
            "test": test_table, "false_alarm_budget_per_week": FA_BUDGET,
            "autoencoder_days_trained": int(ens.ae.n_days_),
        },
    }
    if tune_res is not None:
        metrics["forecast"]["tuning"] = tune_res.head(5).to_dict(orient="records")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (MODELS_DIR / "thresholds.json").write_text(json.dumps(th.as_dict(), indent=2))
    make_samples(hourly)
    log("Done. Reports in reports/, models in models/, demo files in data/sample/")
    return metrics


if __name__ == "__main__":
    main()
