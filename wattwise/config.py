"""Central configuration: paths, model hyper-parameters and default economic settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "raw" / "household_power_consumption.txt"
PROCESSED_FILE = DATA_DIR / "processed" / "hourly.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SAMPLE_DIR = DATA_DIR / "sample"

# Friendly names for the sub-meters of the UCI / Kaggle household dataset.
UCI_CIRCUITS = {
    "Sub_metering_1": "kitchen",          # dishwasher, oven, microwave
    "Sub_metering_2": "laundry",          # washing machine, tumble dryer, fridge, light
    "Sub_metering_3": "water_heater_ac",  # electric water heater and air conditioner
}
CIRCUIT_LABELS = {
    "kitchen": "Kitchen (dishwasher, oven, microwave)",
    "laundry": "Laundry room (washer, dryer, fridge)",
    "water_heater_ac": "Water heater & AC",
    "other": "Other / unmetered (lighting, electronics, standby)",
}

# Chronological split used for training / calibration / honest testing.
TRAIN_END = "2009-12-31 23:00"
CAL_END = "2010-06-30 23:00"  # calibration + threshold tuning window; test = everything after

QUANTILES = (0.1, 0.5, 0.9)  # 80 % prediction interval + median

LGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


@dataclass
class EconomicSettings:
    """Tariff and emission settings used to turn avoidable kWh into money and CO2.

    Defaults are illustrative for an Indian household and are meant to be changed in the app.
    """

    currency: str = "₹"
    flat_rate: float = 8.0  # currency per kWh
    use_tou: bool = False
    peak_rate: float = 10.0
    offpeak_rate: float = 6.0
    peak_hours: tuple = field(default_factory=lambda: tuple(range(18, 22)))  # 18:00–21:59
    offpeak_hours: tuple = field(default_factory=lambda: tuple(list(range(22, 24)) + list(range(0, 6))))
    co2_kg_per_kwh: float = 0.71  # approx. Indian grid average emission factor

    def rate_for_hour(self, hour: int) -> float:
        if not self.use_tou:
            return self.flat_rate
        if hour in self.peak_hours:
            return self.peak_rate
        if hour in self.offpeak_hours:
            return self.offpeak_rate
        return self.flat_rate
