"""Three complementary anomaly detectors and a context-aware ensemble.

1. Contextual residual detector — compares each hour with the forecaster's day-ahead
   expectation, then standardises the residual with robust (median/MAD) statistics learned
   separately for every (hour-of-day x weekday/weekend) context.  A 2 kWh hour at 20:00 on a
   Sunday can be perfectly normal; the same 2 kWh at 03:00 is not.  This is what separates
   *unusual* consumption from *normal daily variation*.
2. Isolation Forest — point anomalies in a multivariate hourly feature space (level,
   deviation from the household's weekly profile, circuit mix, time of day).
3. Day-profile autoencoder — a neural network (MLP, bottleneck) that learns the household's
   typical 24-hour load shapes per circuit; days it cannot reconstruct are shape anomalies,
   and the per-hour reconstruction error localises *when* the day went wrong.

The ensemble requires agreement between detectors (or one overwhelming signal) and labels every
hour as Normal / High-but-normal variation / Unusual-waste / Unusual-low / Data gap.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

EPS = 1e-3

LABEL_NORMAL = "Normal"
LABEL_HIGH_NORMAL = "High but normal variation"
LABEL_WASTE = "Unusual — likely waste"
LABEL_LOW = "Unusual — low"
LABEL_GAP = "Data gap"
LABEL_STANDBY = "Elevated standby load"
LABEL_ORDER = [LABEL_WASTE, LABEL_STANDBY, LABEL_LOW, LABEL_HIGH_NORMAL, LABEL_NORMAL, LABEL_GAP]


def circuit_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("c_") and c != "c_other"]


def build_frame(hourly: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Join readings with expectations and add context columns."""
    f = hourly.copy()
    for c in ("p10", "p50", "p90", "scale"):
        f[c] = expected[c].reindex(f.index)
    # fall back to a robust local scale when the forecaster had too little history
    fallback = f["kwh"].rolling(24 * 14, min_periods=24).median().bfill()
    f["scale"] = f["scale"].fillna(fallback).fillna(f["kwh"].median())
    f["hour"] = f.index.hour
    f["weekend"] = (f.index.dayofweek >= 5).astype(int)
    f["context"] = f["hour"] * 2 + f["weekend"]
    f["resid_n"] = (f["kwh"] - f["p50"]) / f["scale"]
    return f


def _percentile(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.searchsorted(ref_sorted, x, side="right") / max(len(ref_sorted), 1)
    return np.where(np.isnan(x), np.nan, out)


# ------------------------------------------------------------------------ detector 1
class ContextualResidualDetector:
    """Robust z-scores of forecast residuals at 1 h, 3 h and 6 h scales, per (hour, weekday/weekend).

    Short windows catch spikes; longer windows accumulate weak-but-sustained evidence such as
    an air-conditioner running all afternoon, which is invisible hour-by-hour in noisy data.
    Circuit-level contextual z-scores are computed too (used for attribution and by the
    Isolation Forest).
    """

    WINDOWS = (1, 3, 6)

    def _windows(self, f: pd.DataFrame) -> dict:
        r = f["resid_n"]
        return {w: (r if w == 1 else r.rolling(w, center=True, min_periods=(w + 1) // 2).mean()) for w in self.WINDOWS}

    @staticmethod
    def _robust(series: pd.Series, ctx: pd.Series):
        g = series.groupby(ctx)
        med = g.median()
        mad = g.apply(lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))))
        return med, (1.4826 * mad)

    def fit(self, f: pd.DataFrame):
        self.stats_ = {}
        for w, r in self._windows(f).items():
            med, sc = self._robust(r, f["context"])
            self.stats_[w] = (med, sc.clip(lower=0.05 / np.sqrt(w)))
        self.circ_stats_ = {}
        for c in circuit_cols(f):
            v = np.log(f[c] / f["scale"] + 0.02)
            med, sc = self._robust(v, f["context"])
            self.circ_stats_[c] = (med, sc.clip(lower=0.1))
        return self

    def score(self, f: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=f.index)
        for w, r in self._windows(f).items():
            med, sc = self.stats_[w]
            out[f"z{w}"] = (r - f["context"].map(med).fillna(0.0)) / f["context"].map(sc).fillna(sc.median())
        zs = out[[f"z{w}" for w in self.WINDOWS]]
        out["z"] = np.where(zs.max(axis=1) >= -zs.min(axis=1), zs.max(axis=1), zs.min(axis=1))
        out.loc[f["kwh"].isna(), "z"] = np.nan
        for c, (med, sc) in self.circ_stats_.items():
            v = np.log(f[c] / f["scale"] + 0.02)
            out["cz_" + c[2:]] = (v - f["context"].map(med)) / f["context"].map(sc)
        return out


class BaseloadShiftDetector:
    """Day-level detector for 'standby creep': the household's always-on floor jumps up.

    Baseload of a day = mean of its 3 lowest hours.  It is compared with the median of the
    previous 14 days and scaled by the recent day-to-day spread (robust z).
    """

    def score(self, f: pd.DataFrame) -> pd.DataFrame:
        daily = f["kwh"].groupby(f.index.normalize())
        base = daily.apply(lambda v: np.nan if v.count() < 18 else np.sort(v.dropna().to_numpy())[:3].mean())
        base.index = pd.DatetimeIndex(base.index)
        base = base.asfreq("D")
        ref = base.rolling(14, min_periods=5).median().shift(1)
        dev = (base - ref).abs()
        spread = 1.4826 * dev.rolling(28, min_periods=7).median().shift(1)
        spread = np.maximum(spread.fillna(0.0), 0.02 + 0.05 * ref)
        z = (base - ref) / spread
        d = pd.DataFrame({"baseload_kw": base, "baseload_ref_kw": ref, "z_base": z})
        out = d.reindex(f.index.normalize())
        out.index = f.index
        return out


# ------------------------------------------------------------------------ detector 2
class IsolationForestDetector:
    def __init__(self, n_estimators=300, random_state=0):
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=1024, random_state=random_state, n_jobs=-1)

    def _features(self, f: pd.DataFrame) -> pd.DataFrame:
        yn = f["kwh"] / f["scale"]
        how = f.index.dayofweek * 24 + f.index.hour
        prof = how.map(self.profile_).to_numpy() if hasattr(self, "profile_") else np.nan
        X = pd.DataFrame(index=f.index)
        X["vs_profile"] = np.log(yn + EPS) - np.log(np.asarray(prof, float) + EPS)
        for col in ("z1", "z3", "z6"):
            X[col] = f[col].clip(-15, 15)
        for c in [c for c in f.columns if c.startswith("cz_")]:
            X[c] = f[c].clip(-15, 15)
        X["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24)
        X["weekend"] = f["weekend"]
        return X

    def fit(self, f: pd.DataFrame):
        ok = f["kwh"].notna()
        yn = f.loc[ok, "kwh"] / f.loc[ok, "scale"]
        how = f.index[ok].dayofweek * 24 + f.index[ok].hour
        self.profile_ = yn.groupby(how).median()
        X = self._features(f.loc[ok]).dropna()
        self.columns_ = list(X.columns)
        self.model.fit(X)
        self.ref_ = np.sort(-self.model.score_samples(X))
        return self

    def score(self, f: pd.DataFrame) -> pd.DataFrame:
        X = self._features(f)
        ok = X[self.columns_].notna().all(axis=1) & f["kwh"].notna()
        raw = pd.Series(np.nan, index=f.index)
        if ok.any():
            raw[ok] = -self.model.score_samples(X.loc[ok, self.columns_])
        return pd.DataFrame({"if_raw": raw, "p_if": _percentile(self.ref_, raw.to_numpy())}, index=f.index)


# ------------------------------------------------------------------------ detector 3
class DayProfileAutoencoder:
    """Dense autoencoder on 24-hour load profiles (one block of 24 values per channel)."""

    def __init__(self, bottleneck: int | None = None, random_state=0):
        self.bottleneck = bottleneck
        self.random_state = random_state

    def _days(self, f: pd.DataFrame) -> pd.DataFrame:
        chans = ["kwh"] + circuit_cols(f)
        mats = []
        for c in chans:
            v = np.log1p(f[c] / f["scale"])
            m = v.groupby([f.index.normalize(), f["hour"]]).first().unstack()
            m = m.reindex(columns=range(24))
            m.columns = [f"{c}_{h}" for h in range(24)]
            mats.append(m)
        return pd.concat(mats, axis=1)

    def fit(self, f: pd.DataFrame):
        D = self._days(f).dropna()
        self.columns_ = list(D.columns)
        self.n_channels_ = len(self.columns_) // 24
        self.scaler_ = StandardScaler().fit(D)
        Z = self.scaler_.transform(D)
        k = self.bottleneck or max(4, 4 * self.n_channels_)
        width = max(32, 16 * self.n_channels_)
        self.model = MLPRegressor(hidden_layer_sizes=(width, k, width), activation="relu", alpha=1e-3,
                                  learning_rate_init=1e-3, max_iter=800, early_stopping=True,
                                  validation_fraction=0.15, n_iter_no_change=25, random_state=self.random_state)
        self.model.fit(Z, Z)
        err = (self.model.predict(Z) - Z) ** 2
        self.ref_hour_ = np.sort(err.reshape(len(Z), self.n_channels_, 24).sum(axis=1).ravel())
        self.ref_day_ = np.sort(err.mean(axis=1))
        self.n_days_ = len(D)
        return self

    def score(self, f: pd.DataFrame) -> pd.DataFrame:
        D = self._days(f).reindex(columns=self.columns_)
        out = pd.DataFrame(index=f.index, columns=["ae_hour", "ae_day", "p_ae", "p_ae_day"], dtype=float)
        ok = D.notna().all(axis=1)
        if not ok.any():
            return out
        Z = self.scaler_.transform(D[ok])
        err = (self.model.predict(Z) - Z) ** 2
        hour_err = err.reshape(len(Z), self.n_channels_, 24).sum(axis=1)
        day_err = err.mean(axis=1)
        days = D.index[ok]
        he = pd.DataFrame(hour_err, index=days, columns=range(24)).stack()
        key = list(zip(f.index.normalize(), f["hour"]))
        out["ae_hour"] = he.reindex(key).to_numpy()
        out["ae_day"] = pd.Series(day_err, index=days).reindex(f.index.normalize()).to_numpy()
        out["p_ae"] = _percentile(self.ref_hour_, out["ae_hour"].to_numpy())
        out["p_ae_day"] = _percentile(self.ref_day_, out["ae_day"].to_numpy())
        out.loc[f["kwh"].isna(), ["p_ae", "p_ae_day"]] = np.nan
        return out

    def reconstruct_day(self, f: pd.DataFrame, day) -> pd.DataFrame | None:
        """Actual vs autoencoder-reconstructed profile of one day (kWh per hour, total channel)."""
        D = self._days(f).reindex(columns=self.columns_)
        day = pd.Timestamp(day).normalize()
        if day not in D.index or D.loc[[day]].isna().any(axis=1).iloc[0]:
            return None
        Z = self.scaler_.transform(D.loc[[day]])
        rec = self.scaler_.inverse_transform(self.model.predict(Z))[0][:24]
        scale = f.loc[f.index.normalize() == day, "scale"].iloc[0]
        return pd.DataFrame({"actual": np.expm1(D.loc[day].to_numpy()[:24]) * scale,
                             "reconstructed": np.expm1(rec).clip(min=0) * scale}, index=range(24))


# ------------------------------------------------------------------------ ensemble

@dataclass
class Thresholds:
    z: float = 3.0          # multi-scale robust z needed from the residual detector
    p: float = 0.98         # percentile at which Isolation Forest / autoencoder confirm
    z_strong: float = 5.0   # residual evidence strong enough on its own
    z_sustained: float = 4.0  # 6-hour-window evidence strong enough on its own (long, moderate waste)
    z_base: float = 4.0     # baseload-shift z for an 'elevated standby' day
    base_min_kw: float = 0.1  # ...and at least this many kW above the reference

    def as_dict(self):
        return asdict(self)


class EnsembleDetector:
    def __init__(self, thresholds: Thresholds | None = None, random_state=0):
        self.th = thresholds or Thresholds()
        self.residual = ContextualResidualDetector()
        self.baseload = BaseloadShiftDetector()
        self.iforest = IsolationForestDetector(random_state=random_state)
        self.ae = DayProfileAutoencoder(random_state=random_state)

    def fit(self, f: pd.DataFrame):
        self.channels_ = circuit_cols(f)
        self.residual.fit(f)
        g = f.join(self.residual.score(f))
        self.iforest.fit(g)
        self.ae.fit(f)
        return self

    def raw_scores(self, f: pd.DataFrame) -> pd.DataFrame:
        s = self.residual.score(f)
        s = s.join(self.iforest.score(f.join(s))).join(self.ae.score(f)).join(self.baseload.score(f))
        return s

    def label(self, f: pd.DataFrame, s: pd.DataFrame, th: Thresholds | None = None) -> pd.DataFrame:
        th = th or self.th
        z = s["z"].fillna(0.0)
        confirm = (s["p_if"].fillna(0) >= th.p) | (s["p_ae"].fillna(0) >= th.p)
        above = f["kwh"] > f["p90"].fillna(np.inf)
        below = f["kwh"] < f["p10"].fillna(-np.inf)
        z6 = s["z6"].fillna(0.0)
        hi = above & (((z >= th.z) & confirm) | (z >= th.z_strong) | (z6 >= th.z_sustained))
        lo = below & (((z <= -th.z) & confirm) | (z <= -th.z_strong)) & ~hi
        standby = (s["z_base"].fillna(0) >= th.z_base) & ((s["baseload_kw"] - s["baseload_ref_kw"]).fillna(0) >= th.base_min_kw)

        out = s.copy()
        out["votes"] = (z.abs() >= th.z).astype(int) + (s["p_if"].fillna(0) >= th.p).astype(int) + (s["p_ae"].fillna(0) >= th.p).astype(int)
        p_res = 1 - np.exp(-np.clip(z, 0, None) / 2.0)
        out["anomaly_score"] = (0.5 * p_res + 0.25 * s["p_if"].fillna(0) + 0.25 * s["p_ae"].fillna(0)).round(4)
        lab = np.full(len(f), LABEL_NORMAL, dtype=object)
        lab[(above & ~hi).to_numpy()] = LABEL_HIGH_NORMAL
        lab[(standby & ~hi).to_numpy()] = LABEL_STANDBY
        lab[lo.to_numpy()] = LABEL_LOW
        lab[hi.to_numpy()] = LABEL_WASTE
        lab[f["kwh"].isna().to_numpy()] = LABEL_GAP
        out["label"] = lab
        return out

    def detect(self, f: pd.DataFrame) -> pd.DataFrame:
        return self.label(f, self.raw_scores(f))


WASTE_LABELS = (LABEL_WASTE, LABEL_STANDBY)


# ------------------------------------------------------------------------ simple baselines (for evaluation)
def naive_global_threshold(f: pd.DataFrame, ref: pd.DataFrame, pct: float) -> pd.Series:
    """Flag the hours above a global percentile of consumption (what a basic smart-meter alert does)."""
    thr = np.nanpercentile(ref["kwh"], pct)
    return f["kwh"] > thr


def hour_of_day_threshold(f: pd.DataFrame, ref: pd.DataFrame, pct: float) -> pd.Series:
    thr = ref.groupby(ref.index.hour)["kwh"].quantile(pct / 100)
    return f["kwh"] > f.index.hour.map(thr).to_numpy()
