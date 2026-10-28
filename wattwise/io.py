"""Loading meter readings in many formats and turning them into a clean hourly series.

Output of every loader is an hourly DataFrame indexed by timestamp with:
    kwh            total energy used in that hour (kWh)
    c_<circuit>    optional per-circuit energy (kWh), e.g. c_kitchen, c_other
    coverage       share of the hour actually observed (0–1)
    imputed        True when the hour was filled by interpolation
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import UCI_CIRCUITS

VALUE_KINDS = ("energy_kwh", "energy_wh", "power_kw", "power_w")
MIN_COVERAGE = 0.5
MAX_INTERP_GAP_H = 3


@dataclass
class LoadInfo:
    source_format: str
    rows_raw: int
    interval_minutes: float
    value_kind: str
    value_column: str
    circuit_columns: list = field(default_factory=list)
    start: str = ""
    end: str = ""
    hours: int = 0
    missing_hours: int = 0
    imputed_hours: int = 0
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- helpers
def _read_text(src) -> str:
    if isinstance(src, (str, Path)):
        return Path(src).read_text(encoding="utf-8", errors="replace")
    data = src.read() if hasattr(src, "read") else src
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    if hasattr(src, "seek"):
        src.seek(0)
    return data


def _is_uci(header: str) -> bool:
    return "Global_active_power" in header and "Sub_metering_1" in header


def _guess_value_kind(name: str) -> str | None:
    n = name.lower()
    if re.search(r"kwh|kilowatt.?hour", n):
        return "energy_kwh"
    if re.search(r"(^|[^k])wh\b|watt.?hour", n):
        return "energy_wh"
    if re.search(r"\bkw\b|_kw\b|kw$|kilowatt|active_power|power_kw", n):
        return "power_kw"
    if re.search(r"\bw\b|_w$|watts?\b", n):
        return "power_w"
    if re.search(r"energy|consumption|usage|units", n):
        return "energy_kwh"
    if re.search(r"power|load|demand", n):
        return "power_kw"
    return None


def _circuit_name(col: str) -> str:
    """'circuit_kitchen' -> 'kitchen'; UCI 'Sub_metering_3' -> 'water_heater_ac'."""
    if col in UCI_CIRCUITS:
        return UCI_CIRCUITS[col]
    name = re.sub(r"\W+", "_", col.lower()).strip("_")
    name = re.sub(r"^(circuit|sub_?meter(ing)?|appliance)_?", "", name)
    return name or "circuit"


def detect_columns(df: pd.DataFrame) -> dict:
    """Best-effort guess of timestamp / value / circuit columns for a generic CSV."""
    cols = list(df.columns)
    low = {c: c.lower().strip() for c in cols}
    ts = [c for c in cols if re.search(r"timestamp|datetime|date.?time|^time$|^ts$", low[c])]
    date = [c for c in cols if low[c] in ("date", "day")]
    timec = [c for c in cols if low[c] in ("time", "hour")]
    if ts:
        ts_spec = [ts[0]]
    elif date and timec:
        ts_spec = [date[0], timec[0]]
    elif date:
        ts_spec = [date[0]]
    else:
        ts_spec = [cols[0]]

    numeric = [c for c in cols if c not in ts_spec and pd.api.types.is_numeric_dtype(df[c])]
    circuits = [c for c in numeric if re.search(r"circuit|sub.?meter|appliance", low[c])]
    candidates = [c for c in numeric if c not in circuits]
    value, kind = None, None
    for c in candidates:
        k = _guess_value_kind(c)
        if k:
            value, kind = c, k
            break
    if value is None and candidates:
        value = candidates[0]
    return {"timestamp": ts_spec, "value": value, "value_kind": kind, "circuits": circuits}


def _parse_timestamp(df: pd.DataFrame, ts_cols: list) -> pd.DatetimeIndex:
    raw = df[ts_cols[0]].astype(str) if len(ts_cols) == 1 else (
        df[ts_cols[0]].astype(str) + " " + df[ts_cols[1]].astype(str))
    best, best_ok = None, -1
    for dayfirst in (False, True):
        try:
            parsed = pd.to_datetime(raw, dayfirst=dayfirst, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(raw, dayfirst=dayfirst, errors="coerce")
        ok = parsed.notna().mean()
        # prefer the parse that yields a monotone series (catches day/month swaps)
        mono = parsed.dropna().is_monotonic_increasing
        score = ok + (0.5 if mono else 0)
        if score > best_ok:
            best, best_ok = parsed, score
    if getattr(best.dt, "tz", None) is not None:
        best = best.dt.tz_convert(None)
    return pd.DatetimeIndex(best)


def _to_hourly(values: pd.DataFrame, kind: str, interval_min: float) -> pd.DataFrame:
    """Aggregate an irregular/regular series of `kind` readings into hourly kWh."""
    per_hour = max(1.0, 60.0 / interval_min) if interval_min < 60 else 1.0
    if kind.startswith("power"):
        factor = 1.0 if kind == "power_kw" else 1e-3
        hourly = values.resample("h").mean() * factor  # mean kW over an hour = kWh
        counts = values.iloc[:, 0].resample("h").count()
    else:
        factor = 1.0 if kind == "energy_kwh" else 1e-3
        if interval_min > 60:  # coarser than hourly: spread evenly over the hours it covers
            spread = values.resample("h").asfreq()
            n = int(round(interval_min / 60))
            hourly = spread.bfill(limit=n - 1) / n * factor
            counts = values.iloc[:, 0].resample("h").count().replace(0, np.nan).bfill(limit=n - 1).fillna(0)
            per_hour = 1.0
        else:
            sums = values.resample("h").sum(min_count=1)
            counts = values.iloc[:, 0].resample("h").count()
            hourly = sums.mul(per_hour / counts.replace(0, np.nan), axis=0) * factor
    coverage = (counts / per_hour).clip(upper=1.0)
    hourly["coverage"] = coverage
    return hourly


def _finalize(hourly: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    hourly = hourly.asfreq("h")
    hourly["coverage"] = hourly["coverage"].fillna(0.0)
    value_cols = [c for c in hourly.columns if c == "kwh" or c.startswith("c_")]
    low_cov = hourly["coverage"] < MIN_COVERAGE
    hourly.loc[low_cov, value_cols] = np.nan
    missing_before = int(hourly["kwh"].isna().sum())
    filled = hourly[value_cols].interpolate(method="time", limit=MAX_INTERP_GAP_H, limit_area="inside")
    hourly["imputed"] = hourly["kwh"].isna() & filled["kwh"].notna()
    hourly[value_cols] = filled.clip(lower=0)
    missing_after = int(hourly["kwh"].isna().sum())
    return hourly, missing_after, missing_before - missing_after


# --------------------------------------------------------------------------- public loaders
def load_uci(src) -> tuple[pd.DataFrame, LoadInfo]:
    """Load the UCI / Kaggle 'Individual household electric power consumption' file (minute data)."""
    buf = src if isinstance(src, (str, Path)) else io.StringIO(_read_text(src))
    raw = pd.read_csv(buf, sep=";", na_values=["?", ""], low_memory=False,
                      dtype={"Date": str, "Time": str})
    ts = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    raw.index = ts
    raw = raw.drop(columns=["Date", "Time"]).astype(float)
    df = pd.DataFrame(index=raw.index)
    df["kwh_min"] = raw["Global_active_power"] / 60.0  # kW over 1 minute -> kWh
    for src_col, name in UCI_CIRCUITS.items():
        df[f"c_{name}"] = raw[src_col] / 1000.0  # Wh -> kWh
    df["c_other"] = (df["kwh_min"] - df[[f"c_{n}" for n in UCI_CIRCUITS.values()]].sum(axis=1, min_count=3)).clip(lower=0)
    counts = df["kwh_min"].resample("h").count()
    hourly = df.resample("h").sum(min_count=1).mul(60.0 / counts.replace(0, np.nan), axis=0)
    hourly = hourly.rename(columns={"kwh_min": "kwh"})
    hourly["coverage"] = counts / 60.0
    hourly, missing, imputed = _finalize(hourly)
    info = LoadInfo(
        source_format="UCI/Kaggle household (minute-level)", rows_raw=len(raw), interval_minutes=1.0,
        value_kind="power_kw", value_column="Global_active_power",
        circuit_columns=[c for c in hourly.columns if c.startswith("c_")],
        start=str(hourly.index[0]), end=str(hourly.index[-1]), hours=len(hourly),
        missing_hours=missing, imputed_hours=imputed,
    )
    return hourly, info


def load_generic(src, timestamp=None, value=None, value_kind=None, circuits=None) -> tuple[pd.DataFrame, LoadInfo]:
    """Load any CSV with a timestamp and a consumption column (energy or power, any interval)."""
    text = _read_text(src)
    sep = ";" if text.split("\n", 1)[0].count(";") > text.split("\n", 1)[0].count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, na_values=["?", "", "NA", "null"])
    guess = detect_columns(df)
    ts_cols = [timestamp] if isinstance(timestamp, str) else (timestamp or guess["timestamp"])
    value = value or guess["value"]
    if value is None:
        raise ValueError("Could not find a numeric consumption column. Pick it manually.")
    value_kind = value_kind or guess["value_kind"] or "energy_kwh"
    circuits = guess["circuits"] if circuits is None else circuits
    notes = []
    if guess["value_kind"] is None and value_kind == "energy_kwh":
        notes.append(f"Unit of '{value}' not obvious from its name; assumed kWh per interval.")

    idx = _parse_timestamp(df, ts_cols)
    keep = idx.notna()
    data = df.loc[keep, [value] + list(circuits)].apply(pd.to_numeric, errors="coerce")
    data.index = idx[keep]
    data = data[~data.index.duplicated(keep="first")].sort_index()
    if len(data) < 48:
        raise ValueError("Need at least 48 readings to analyse.")
    interval = float(pd.Series(data.index).diff().dt.total_seconds().median() / 60.0)
    renamed = data.rename(columns={value: "kwh", **{c: "c_" + _circuit_name(c) for c in circuits}})
    hourly = _to_hourly(renamed, value_kind, interval)
    if circuits:
        circ = [c for c in hourly.columns if c.startswith("c_")]
        hourly["c_other"] = (hourly["kwh"] - hourly[circ].sum(axis=1, min_count=1)).clip(lower=0)
    hourly, missing, imputed = _finalize(hourly)
    info = LoadInfo(
        source_format="Generic CSV", rows_raw=len(df), interval_minutes=round(interval, 2),
        value_kind=value_kind, value_column=value,
        circuit_columns=[c for c in hourly.columns if c.startswith("c_")],
        start=str(hourly.index[0]), end=str(hourly.index[-1]), hours=len(hourly),
        missing_hours=missing, imputed_hours=imputed, notes=notes,
    )
    return hourly, info


def load_any(src, **kwargs) -> tuple[pd.DataFrame, LoadInfo]:
    """Dispatch to the right loader based on the file header."""
    text_head = _read_text(src)[:500] if not isinstance(src, (str, Path)) else Path(src).open(errors="replace").read(500)
    if _is_uci(text_head):
        return load_uci(src)
    return load_generic(src, **kwargs)


def to_generic_csv(hourly: pd.DataFrame) -> pd.DataFrame:
    """Export an hourly frame in the simple upload format (timestamp, kwh, circuit_*)."""
    out = pd.DataFrame({"timestamp": hourly.index.strftime("%Y-%m-%d %H:%M"), "kwh": hourly["kwh"].round(4).values})
    for c in hourly.columns:
        if c.startswith("c_") and c != "c_other":
            out["circuit_" + c[2:]] = hourly[c].round(4).values
    return out
