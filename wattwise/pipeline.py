"""End-to-end analysis of an uploaded meter: expectations, anomalies, events, savings, forecast."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import joblib
import pandas as pd

from .anomaly import EnsembleDetector, Thresholds, build_frame, circuit_cols
from .config import MODELS_DIR, EconomicSettings
from .forecast import Forecaster
from .io import LoadInfo
from . import savings as sv

PRETRAINED_CIRCUITS = ["c_kitchen", "c_laundry", "c_water_heater_ac"]
MIN_DAYS = 14
MIN_DAYS_OWN_DETECTOR = 28
MIN_DAYS_RETRAIN = 120


@dataclass
class AnalysisResult:
    frame: pd.DataFrame
    events: pd.DataFrame
    standby: pd.DataFrame
    summary: dict
    recommendations: list
    forecast: pd.DataFrame
    info: LoadInfo
    detector: EnsembleDetector
    forecaster: Forecaster
    notes: list = field(default_factory=list)


def load_models(models_dir=MODELS_DIR) -> dict:
    try:
        return {
            "forecaster": joblib.load(models_dir / "forecaster.joblib"),
            "detector_with_circuits": joblib.load(models_dir / "detector_with_circuits.joblib"),
            "detector_total_only": joblib.load(models_dir / "detector_total_only.joblib"),
        }
    except Exception as e:  # missing files or a scikit-learn/LightGBM version mismatch
        raise RuntimeError(
            f"Could not load the saved models ({e}). Rebuild them for your library versions with "
            "`python train.py` (about 4 minutes, uses data/processed/hourly.parquet).") from e


def analyze(hourly: pd.DataFrame, info: LoadInfo, models: dict, econ: EconomicSettings | None = None,
            country: str | None = "IN", detector_mode: str = "auto", retrain_forecaster: bool = False,
            self_calibrate: bool = True, horizon_days: int = 7, flex_share: float = 0.15) -> AnalysisResult:
    econ = econ or EconomicSettings()
    notes = []
    y = hourly["kwh"]
    days = y.notna().sum() / 24
    if days < MIN_DAYS:
        raise ValueError(f"Need at least {MIN_DAYS} days of readings; got {days:.1f}.")

    # ------------------------------------------------------------ forecaster
    fc: Forecaster = copy.deepcopy(models["forecaster"])
    fc.country = country
    if retrain_forecaster and days >= MIN_DAYS_RETRAIN:
        cut = y.last_valid_index() - pd.Timedelta(days=42)
        fc = Forecaster(country, params=fc.params).fit(y, end=cut)
        fc.margins = dict(models["forecaster"].margins)
        notes.append(f"Forecaster re-trained on this meter's data up to {cut:%d %b %Y}; the last 6 weeks were kept for calibration.")
    if self_calibrate and fc.recalibrate(y):
        notes.append("Prediction bands self-calibrated on this meter's last 6 weeks of forecast errors (rolling conformal).")
    elif self_calibrate:
        notes.append("Not enough history to self-calibrate prediction bands; using the bands calibrated on the training household.")

    expected = fc.predict_insample(y)
    f = build_frame(hourly, expected)

    # ------------------------------------------------------------ detector
    th = models["detector_total_only"].th
    has_pre_circuits = set(circuit_cols(f)) == set(PRETRAINED_CIRCUITS)
    mode = detector_mode
    if mode == "auto":
        mode = "own" if days >= MIN_DAYS_OWN_DETECTOR else "pretrained"
    if mode == "own":
        det = EnsembleDetector(Thresholds(**th.as_dict())).fit(f)
        notes.append(f"Anomaly detectors learned this household's own normal pattern from {days:.0f} days of data.")
    else:
        if has_pre_circuits:
            det = models["detector_with_circuits"]
        else:
            det = models["detector_total_only"]
            if circuit_cols(f):
                notes.append("Circuit columns differ from the training household; pretrained detector uses total consumption only.")
                f = f.drop(columns=[c for c in f.columns if c.startswith("c_")])
        notes.append("Using detectors pretrained on the Kaggle/UCI household.")
    scores = det.detect(f)
    frame = f.join(scores)

    # ------------------------------------------------------------ savings, events, forecast
    events = sv.group_events(frame, det, econ)
    standby = sv.standby_analysis(frame)
    summary = sv.summarize(frame, events, standby, econ, flex_share)
    recs = sv.recommendations(events, standby, summary, econ)
    fcst = fc.forecast(y, days=horizon_days)
    if info.missing_hours:
        notes.append(f"{info.missing_hours} hours had no usable readings (shown as data gaps); {info.imputed_hours} short gaps were interpolated.")
    return AnalysisResult(frame, events, standby, summary, recs, fcst, info, det, fc, notes + list(info.notes))
