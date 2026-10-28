"""Unit tests for loading, leakage-free features, forecasting and waste injection (python -m pytest tests/)."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wattwise.features import FEATURES, build_features  # noqa: E402
from wattwise.injection import inject  # noqa: E402
from wattwise.io import load_any, load_generic  # noqa: E402


def synthetic(days=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    daily = 0.3 + 0.8 * np.exp(-((idx.hour - 20) ** 2) / 8) + 0.5 * np.exp(-((idx.hour - 8) ** 2) / 4)
    return pd.Series(daily + rng.gamma(2, 0.05, len(idx)), index=idx, name="kwh")


def test_features_have_no_lookahead():
    y = synthetic()
    day = pd.Timestamp("2024-02-10")
    f1 = build_features(y, None).loc[day: day + pd.Timedelta(hours=23), FEATURES]
    y2 = y.copy()
    y2.loc[day: day + pd.Timedelta(days=5)] *= 10  # change the target day and the future
    f2 = build_features(y2, None).loc[day: day + pd.Timedelta(hours=23), FEATURES]
    pd.testing.assert_frame_equal(f1, f2)


def test_generic_loader_units_and_interval():
    y = synthetic(20)
    q = y.resample("15min").ffill()
    df = pd.DataFrame({"Reading Time": q.index.strftime("%d-%m-%Y %H:%M"), "Load (W)": q.values * 1000})
    hourly, info = load_any(io.BytesIO(df.to_csv(index=False).encode()))
    assert info.value_kind == "power_w" and info.interval_minutes == 15
    common = hourly.index.intersection(y.index)[:-1]
    assert np.allclose(hourly.loc[common, "kwh"], y.loc[common], rtol=1e-6)


def test_energy_per_interval_is_summed():
    y = synthetic(16)
    q = (y.resample("30min").ffill() / 2)  # kWh per half hour
    df = pd.DataFrame({"timestamp": q.index.astype(str), "kwh": q.values, "circuit_geyser": q.values * 0.3})
    hourly, info = load_generic(io.BytesIO(df.to_csv(index=False).encode()))
    assert info.circuit_columns == ["c_geyser", "c_other"]
    common = hourly.index.intersection(y.index)[:-1]
    assert np.allclose(hourly.loc[common, "kwh"], y.loc[common], rtol=1e-6)


def test_injection_labels_match_added_energy():
    y = synthetic(40)
    h = pd.DataFrame({"kwh": y})
    hi, lab, ev = inject(h, n_per_type=2, seed=3)
    added = (hi["kwh"] - h["kwh"]).sum()
    assert added == pytest.approx(ev["kwh_added"].sum(), rel=0.01)
    assert set(lab[lab != ""].unique()) <= set(ev["type"])


@pytest.mark.skipif(not (Path(__file__).resolve().parent.parent / "models" / "forecaster.joblib").exists(), reason="run train.py first")
def test_forecast_shapes_and_bands():
    from wattwise.pipeline import load_models
    fc = load_models()["forecaster"]
    fc.country = None
    out = fc.forecast(synthetic(50), days=3)
    assert len(out) == 72
    assert (out["p10"] <= out["p50"]).all() and (out["p50"] <= out["p90"]).all()
    daily = fc.daily_totals(out)
    assert len(daily) == 3 and (daily["p10"] < daily["p50"]).all()
