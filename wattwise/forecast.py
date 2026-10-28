"""Probabilistic day-ahead and multi-day demand forecasting.

Model: LightGBM gradient-boosted trees trained with the quantile (pinball) loss for the
10th, 50th and 90th percentiles on scale-normalised targets.  Multi-day forecasts are made
recursively, one day at a time.  Prediction intervals are then calibrated with
*conformalized quantile regression* (CQR) separately for every lead day, so the 80 % band
really contains ~80 % of future readings even far ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .config import LGB_PARAMS, QUANTILES
from .features import FEATURES, build_features, target

HISTORY_DAYS_NEEDED = 35  # lags go back 4 weeks + scale window


class Forecaster:
    def __init__(self, country: str | None = "FR", quantiles=QUANTILES, params: dict | None = None):
        self.country = country
        self.quantiles = tuple(quantiles)
        self.params = dict(params or LGB_PARAMS)
        self.models: dict[float, LGBMRegressor] = {}
        self.margins: dict[int, float] = {}
        self.alpha = 1.0 - (self.quantiles[-1] - self.quantiles[0])
        self.feature_importance_: pd.Series | None = None

    # ------------------------------------------------------------------ training
    def fit(self, y: pd.Series, end=None) -> "Forecaster":
        feats = build_features(y, self.country)
        tgt = target(y, feats)
        mask = tgt.notna() & feats["scale"].notna() & feats["lag24"].notna()
        if end is not None:
            mask &= feats.index <= pd.Timestamp(end)
        X, t = feats.loc[mask, FEATURES], tgt[mask]
        for q in self.quantiles:
            m = LGBMRegressor(objective="quantile", alpha=q, **self.params)
            m.fit(X, t)
            self.models[q] = m
        mid = self.models[self.quantiles[len(self.quantiles) // 2]]
        self.feature_importance_ = pd.Series(mid.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
        self.n_train_ = int(mask.sum())
        return self

    def _predict_norm(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([self.models[q].predict(X[FEATURES]) for q in self.quantiles])
        return np.sort(preds, axis=1)  # no quantile crossing

    # ------------------------------------------------------------------ in-sample "expected" values
    def predict_insample(self, y: pd.Series) -> pd.DataFrame:
        """Day-ahead expectation for every historical hour (uses only info up to the previous day)."""
        feats = build_features(y, self.country)
        ok = feats["scale"].notna() & feats["lag24"].notna()
        out = pd.DataFrame(index=feats.index, columns=["p10", "p50", "p90", "scale"], dtype=float)
        if ok.any():
            p = self._predict_norm(feats.loc[ok])
            sc = feats.loc[ok, "scale"].to_numpy()[:, None]
            out.loc[ok, ["p10", "p50", "p90"]] = p * sc
            out.loc[ok, "scale"] = sc[:, 0]
        return self._apply_margin(out, lead=1)

    # ------------------------------------------------------------------ recursive forecasting
    def _recursive(self, hist: pd.Series, days: int) -> pd.DataFrame:
        hist = hist.asfreq("h")
        last_obs = hist.last_valid_index()
        hist = hist.loc[:last_obs]
        start_day = last_obs.normalize() + (pd.Timedelta(days=1) if last_obs.hour == 23 else pd.Timedelta(0))
        win_start = start_day - pd.Timedelta(days=HISTORY_DAYS_NEEDED)
        full_idx = pd.date_range(win_start, start_day + pd.Timedelta(days=days) - pd.Timedelta(hours=1), freq="h")
        ext = hist.reindex(full_idx)
        rows = []
        for k in range(days):
            d0 = start_day + pd.Timedelta(days=k)
            d1 = d0 + pd.Timedelta(hours=23)
            feats = build_features(ext.loc[:d1], self.country).loc[d0:d1]
            p = self._predict_norm(feats) * feats["scale"].to_numpy()[:, None]
            day = pd.DataFrame(p, index=feats.index, columns=["p10", "p50", "p90"])
            day["scale"] = feats["scale"].to_numpy()
            day["lead"] = k + 1
            rows.append(day)
            fill = ext.loc[d0:d1].isna()
            ext.loc[d0:d1] = ext.loc[d0:d1].where(~fill, day["p50"])
        out = pd.concat(rows)
        out.attrs["cutoff"] = last_obs
        return out

    def _apply_margin(self, df: pd.DataFrame, lead=None) -> pd.DataFrame:
        df = df.copy()
        if lead is not None:
            m = self.margins.get(lead, 0.0)
        else:
            default = max(self.margins.values()) if self.margins else 0.0
            m = df["lead"].map(lambda l: self.margins.get(int(l), default)).to_numpy()
        df["p10"] = (df["p10"] - m * df["scale"]).clip(lower=0)
        df["p90"] = df["p90"] + m * df["scale"]
        df["p50"] = df["p50"].clip(lower=df["p10"], upper=df["p90"])
        return df

    def forecast(self, y: pd.Series, days: int = 7) -> pd.DataFrame:
        """Calibrated forecast for the hours after the last observation."""
        rec = self._recursive(y, days)
        rec = self._apply_margin(rec)
        out = rec[rec.index > rec.attrs["cutoff"]].copy()
        out.attrs = {}
        return out

    # ------------------------------------------------------------------ backtesting & calibration
    def backtest(self, y: pd.Series, start, end, horizon: int = 7, step: int = 1, calibrated=True) -> pd.DataFrame:
        y = y.asfreq("h")
        origins = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq=f"{step}D")
        parts = []
        for o in origins:
            hist = y.loc[: o - pd.Timedelta(hours=1)]
            if hist.loc[o - pd.Timedelta(days=7): ].notna().sum() < 24 * 5:
                continue  # skip origins right after long outages
            rec = self._recursive(hist, horizon)
            rec["origin"] = o
            parts.append(rec)
        bt = pd.concat(parts)
        bt["actual"] = y.reindex(bt.index).to_numpy()
        if calibrated:
            bt = self._apply_margin(bt)
        return bt.dropna(subset=["actual"])

    @staticmethod
    def conformity(bt: pd.DataFrame) -> pd.Series:
        return np.maximum(bt["p10"] - bt["actual"], bt["actual"] - bt["p90"]) / bt["scale"]

    def _margin(self, scores) -> float:
        s = np.asarray(scores, float)
        s = s[~np.isnan(s)]
        n = len(s)
        if n == 0:
            return 0.0
        return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)))

    def calibrate(self, y: pd.Series, start, end, horizon: int = 7, step: int = 1) -> pd.DataFrame:
        """Conformalized quantile regression: per-lead-day additive margins (normalised units)."""
        bt = self.backtest(y, start, end, horizon=horizon, step=step, calibrated=False)
        score = self.conformity(bt)
        for lead, s in score.groupby(bt["lead"]):
            self.margins[int(lead)] = self._margin(s)
        self._calibrate_daily(bt)
        self.calibration_info_ = {"start": str(start), "end": str(end), "n_scores": int(score.notna().sum())}
        return bt

    def _calibrate_daily(self, bt: pd.DataFrame):
        """Daily-total bands. Hourly quantiles do not add up, so the spread of daily totals is
        learned directly: P10/P90 of log(actual total / forecast total) for each lead day."""
        g = bt.assign(day=bt.index.normalize()).groupby(["origin", "lead", "day"])
        d = g.agg(actual=("actual", "sum"), p50=("p50", "sum"), n=("actual", "count"))
        d = d[(d.n == 24) & (d.p50 > 0) & (d.actual > 0)]
        lr = np.log(d.actual / d.p50)
        q = self.quantiles
        self.daily_q_ = {int(l): (float(v.quantile(q[0])), float(v.quantile(q[-1]))) for l, v in lr.groupby(level="lead")}

    def daily_totals(self, fc: pd.DataFrame, complete_only: bool = True) -> pd.DataFrame:
        """Daily totals of a forecast with properly calibrated P10–P90 ranges."""
        dq = getattr(self, "daily_q_", None) or {l: (-0.45, 0.35) for l in range(1, 15)}
        g = fc.groupby(fc.index.normalize())
        d = pd.DataFrame({"p50": g["p50"].sum(), "hours": g["p50"].count(), "lead": g["lead"].min()})
        if complete_only:
            d = d[d.hours == 24]
        far = max(dq)
        lo = d["lead"].map(lambda l: dq.get(int(l), dq[far])[0])
        hi = d["lead"].map(lambda l: dq.get(int(l), dq[far])[1])
        d["p10"], d["p90"] = d["p50"] * np.exp(lo), d["p50"] * np.exp(hi)
        return d[["p10", "p50", "p90", "hours", "lead"]]

    @staticmethod
    def period_total(daily: pd.DataFrame) -> tuple[float, float, float]:
        """Total over several days; day-level errors combined in quadrature (≈ independent days)."""
        mid = float(daily["p50"].sum())
        lo = mid - float(np.sqrt(((daily["p50"] - daily["p10"]) ** 2).sum()))
        hi = mid + float(np.sqrt(((daily["p90"] - daily["p50"]) ** 2).sum()))
        return max(lo, 0.0), mid, hi

    def recalibrate(self, y: pd.Series, window_days: int = 42, step: int = 2, min_origins: int = 8) -> bool:
        """Self-calibration on the most recent `window_days` of *this* meter (rolling conformal).

        Bands are re-fitted to the forecast errors this model actually makes on the uploaded
        household, which keeps coverage close to the target under seasonal or behavioural drift.
        Returns False (and keeps the stored margins) when there is not enough history.
        """
        y = y.asfreq("h")
        last = y.last_valid_index().normalize()
        first_ok = y.first_valid_index().normalize() + pd.Timedelta(days=HISTORY_DAYS_NEEDED)
        start = max(first_ok, last - pd.Timedelta(days=window_days))
        origins = pd.date_range(start, last - pd.Timedelta(days=1), freq=f"{step}D")
        if len(origins) < min_origins:
            return False
        self.calibrate(y, origins[0], origins[-1], horizon=7, step=step)
        return True


# ---------------------------------------------------------------------- metrics & baselines
def point_metrics(actual, pred) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    ok = ~(np.isnan(a) | np.isnan(p))
    a, p = a[ok], p[ok]
    err = p - a
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err) / np.maximum(a, 1e-6)) * 100),
        "WAPE_%": float(np.sum(np.abs(err)) / np.sum(a) * 100),
        "n": int(ok.sum()),
    }


def interval_metrics(actual, lo, hi, med, quantiles=QUANTILES) -> dict:
    a, lo, hi, med = (np.asarray(v, float) for v in (actual, lo, hi, med))
    ok = ~(np.isnan(a) | np.isnan(lo) | np.isnan(hi) | np.isnan(med))
    a, lo, hi, med = a[ok], lo[ok], hi[ok], med[ok]

    def pinball(q, pred):
        d = a - pred
        return np.mean(np.maximum(q * d, (q - 1) * d))

    return {
        "coverage_80_%": float(np.mean((a >= lo) & (a <= hi)) * 100),
        "mean_width_kWh": float(np.mean(hi - lo)),
        "pinball_avg": float(np.mean([pinball(quantiles[0], lo), pinball(quantiles[1], med), pinball(quantiles[2], hi)])),
    }


def naive_baselines(y: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Classic day-ahead baselines evaluated on `index` (all use information available the day before)."""
    y = y.asfreq("h")
    out = pd.DataFrame(index=index)
    out["Persistence (same hour yesterday)"] = y.shift(24).reindex(index)
    out["Seasonal naive (same hour last week)"] = y.shift(168).reindex(index)
    out["4-week same-hour median"] = pd.concat([y.shift(168 * j) for j in range(1, 5)], axis=1).median(axis=1).reindex(index)
    return out
