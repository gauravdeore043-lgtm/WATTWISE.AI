"""Feature engineering for day-ahead load forecasting.

Every feature for an hour on day D only uses readings up to 23:00 of day D-1, so the
same function serves training, in-sample "expected" values and recursive multi-day forecasts
without leakage.  Values are divided by a trailing 28-day household scale so one model
transfers across meters of very different size (scale-invariant global model).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

FEATURES = [
    "hour", "dow", "is_weekend", "is_holiday", "doy_sin", "doy_cos",
    "lag24", "lag25", "lag48", "lag72", "lag168", "lag336",
    "same_hour_7d_mean", "same_hour_4w_median",
    "prev_day_mean", "prev_day_max", "prev_day_min", "prev_day_std",
    "prev_7d_mean", "last_obs",
]


@lru_cache(maxsize=64)
def _holiday_dates(country: str, year: int) -> frozenset:
    try:
        import holidays
        return frozenset(holidays.country_holidays(country, years=year).keys())
    except Exception:  # unknown country code or package missing
        return frozenset()


def holiday_flags(index: pd.DatetimeIndex, country: str | None) -> np.ndarray:
    if not country:
        return np.zeros(len(index), dtype=int)
    days = index.normalize()
    dates = set()
    for yr in np.unique(index.year):
        dates |= _holiday_dates(country, int(yr))
    if not dates:
        return np.zeros(len(index), dtype=int)
    return days.isin(pd.DatetimeIndex(sorted(dates))).astype(int)


def daily_scale(y: pd.Series) -> pd.Series:
    """Trailing 28-day mean hourly kWh known at the start of each day (shifted by one day)."""
    daily = y.resample("D").mean()
    scale = daily.rolling(28, min_periods=3).mean().shift(1)
    scale = scale.fillna(daily.expanding(min_periods=1).mean().shift(1))
    return scale


def build_features(y: pd.Series, country: str | None = "FR") -> pd.DataFrame:
    """Build the feature matrix for every hour of `y` (hourly, regular index, NaN allowed)."""
    y = y.asfreq("h")
    idx = y.index
    day = idx.normalize()
    scale_d = daily_scale(y)
    scale = scale_d.reindex(day).to_numpy()
    s = pd.Series(scale, index=idx)

    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["is_holiday"] = holiday_flags(idx, country)
    doy = idx.dayofyear.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    for k in (24, 25, 48, 72, 168, 336):
        f[f"lag{k}"] = y.shift(k) / s
    f["same_hour_7d_mean"] = pd.concat([y.shift(24 * j) for j in range(1, 8)], axis=1).mean(axis=1) / s
    f["same_hour_4w_median"] = pd.concat([y.shift(168 * j) for j in range(1, 5)], axis=1).median(axis=1) / s

    daily = y.resample("D")
    prev = pd.DataFrame({
        "prev_day_mean": daily.mean(), "prev_day_max": daily.max(),
        "prev_day_min": daily.min(), "prev_day_std": daily.std(),
    }).shift(1)
    prev["prev_7d_mean"] = y.resample("D").mean().rolling(7, min_periods=3).mean().shift(1)
    last = y.resample("D").last().shift(1)
    prev["last_obs"] = last
    prev = prev.reindex(day)
    prev.index = idx
    for c in prev.columns:
        f[c] = prev[c] / s
    f["scale"] = s
    return f


def target(y: pd.Series, feats: pd.DataFrame) -> pd.Series:
    return y.asfreq("h") / feats["scale"]
