"""Shipment preparation and feature engineering module.

Provides shared constants, cleaning routines, and feature construction
used across all modelling stages and scoring scripts.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(
    raw: pd.DataFrame, medians: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Clean raw shipment records and impute missing numeric values using medians.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw shipment records export.
    medians : dict, optional
        Pre-computed imputation medians with keys 'distance_km',
        'parcel_weight_kg', and 'declared_value_inr'. If None, medians are
        computed from non-missing values after deduplication and sentinel removal.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        Cleaned shipment DataFrame and the dictionary of imputation medians.
    """
    # 1. Drop exact duplicate rows
    df = raw.drop_duplicates().copy()

    # 2. Standardize service_type: strip whitespace and convert to title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # 3. Handle missing payment_mode by filling with 'Unknown'
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. Clean declared_value: remove rupee symbol and commas, convert to numeric INR, drop text column
    val_clean = (
        df["declared_value"]
        .astype(str)
        .str.replace("\u20b9", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["declared_value_inr"] = pd.to_numeric(val_clean, errors="coerce")
    df = df.drop(columns=["declared_value"])

    # 5. Convert dispatch_time to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # 6. Treat parcel_weight_kg > 100 as scanner-fault sentinel (set to NaN)
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. Compute medians on non-missing values if not supplied, then impute
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }

    for col in ["distance_km", "parcel_weight_kg", "declared_value_inr"]:
        df[col] = df[col].fillna(medians[col])

    return df, medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attach hub metadata and engineer all operational and temporal features.

    Parameters
    ----------
    clean : pd.DataFrame
        Cleaned shipment records (without missing feature values).
    hubs : pd.DataFrame
        Hub metadata containing hub_id, hub_city, region, hub_tier, hub_daily_capacity.

    Returns
    -------
    pd.DataFrame
        Shipment DataFrame enriched with engineered features and hub metadata.
    """
    # Left merge with hubs metadata
    df = clean.merge(hubs, on="hub_id", how="left")

    # Engineered operational features
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
