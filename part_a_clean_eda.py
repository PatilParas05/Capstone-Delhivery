"""Part A: Data Foundation and Delay Evidence.

Loads raw shipment data, cleans and imputes missing values, engineers features,
validates quality checks against reference benchmarks, and produces delay evidence
and correlation summaries.
"""

import os
import numpy as np
import pandas as pd

from src.shipment_prep import (
    FEATURES,
    clean_shipments,
    build_features,
)


def run_part_a():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Load raw files
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # Metrics before cleaning (measured on deduped data before imputation)
    raw_rows = len(raw)
    dupes_removed = raw.duplicated().sum()
    raw_dedup = raw.drop_duplicates()
    rows_after_dedup = len(raw_dedup)

    missing_dist_before = int(raw_dedup["distance_km"].isna().sum())
    missing_weight_before = int(raw_dedup["parcel_weight_kg"].isna().sum())
    sentinel_weight_count = int((raw_dedup["parcel_weight_kg"] > 100).sum())
    missing_declared_before = int(raw_dedup["declared_value"].isna().sum())
    missing_payment_before = int(raw_dedup["payment_mode"].isna().sum())
    service_spellings_before = int(raw_dedup["service_type"].nunique())

    # Clean shipments
    clean_df, medians = clean_shipments(raw)
    service_spellings_after = int(clean_df["service_type"].nunique())

    # Build features
    model_df = build_features(clean_df, hubs)

    # Target variable: is_delayed (delivery_hours > promised_hours)
    model_df["is_delayed"] = (
        model_df["delivery_hours"] > model_df["promised_hours"]
    ).astype(int)

    # Write modelling table
    model_df.to_csv("data/modelling_table.csv", index=False)

    # Quality and distribution checks
    rows_after_hub_merge = len(model_df)
    nulls_features_target = int(
        model_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
    )
    delayed_count = int(model_df["is_delayed"].sum())
    delay_rate = float(model_df["is_delayed"].mean())

    # Quartiles & IQR on delivery_hours (computed unrounded, rounded when written)
    q1 = float(model_df["delivery_hours"].quantile(0.25))
    q3 = float(model_df["delivery_hours"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(
        (
            (model_df["delivery_hours"] < lower_bound)
            | (model_df["delivery_hours"] > upper_bound)
        ).sum()
    )

    declared_mean = float(model_df["declared_value_inr"].mean())
    declared_median = float(model_df["declared_value_inr"].median())
    log_declared_mean = float(model_df["log_declared_value"].mean())
    log_declared_median = float(model_df["log_declared_value"].median())

    checks_data = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", dupes_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_dist_before),
        ("missing_parcel_weight_before", missing_weight_before),
        ("sentinel_weight_rows", sentinel_weight_count),
        ("missing_declared_value_before", missing_declared_before),
        ("missing_payment_mode_before", missing_payment_before),
        ("service_type_spellings_before", service_spellings_before),
        ("service_type_spellings_after", service_spellings_after),
        ("median_distance_km", round(medians["distance_km"], 4)),
        ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 4)),
        ("median_declared_value_inr", round(medians["declared_value_inr"], 4)),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_features_target),
        ("delayed_count", delayed_count),
        ("delay_rate", round(delay_rate, 4)),
        ("hours_q1", round(q1, 4)),
        ("hours_q3", round(q3, 4)),
        ("hours_lower_bound", round(lower_bound, 4)),
        ("hours_upper_bound", round(upper_bound, 4)),
        ("hours_outlier_count", outlier_count),
        ("declared_value_mean_inr", round(declared_mean, 4)),
        ("declared_value_median_inr", round(declared_median, 4)),
        ("log_declared_value_mean", round(log_declared_mean, 4)),
        ("log_declared_value_median", round(log_declared_median, 4)),
    ]

    int_metrics = {
        "raw_rows",
        "duplicate_rows_removed",
        "rows_after_dedup",
        "missing_distance_km_before",
        "missing_parcel_weight_before",
        "sentinel_weight_rows",
        "missing_declared_value_before",
        "missing_payment_mode_before",
        "service_type_spellings_before",
        "service_type_spellings_after",
        "rows_after_hub_merge",
        "nulls_in_features_and_target",
        "delayed_count",
        "hours_outlier_count",
    }

    formatted_checks = []
    for m, v in checks_data:
        if m in int_metrics:
            formatted_checks.append((m, str(int(v))))
        else:
            formatted_checks.append((m, f"{round(float(v), 4):g}"))

    checks_df = pd.DataFrame(formatted_checks, columns=["metric", "value"])
    checks_df.to_csv("outputs/part_a_checks.csv", index=False)

    # Delay evidence by group
    model_df["distance_band"] = pd.cut(
        model_df["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
    )

    group_rows = []
    for dim in ["service_type", "hub_tier", "distance_band"]:
        grouped = (
            model_df.groupby(dim, observed=False)
            .agg(
                shipments=("delivery_hours", "count"),
                delayed=("is_delayed", "sum"),
                delay_rate=("is_delayed", "mean"),
                mean_hours=("delivery_hours", "mean"),
            )
            .reset_index()
            .rename(columns={dim: "group"})
        )
        grouped["dimension"] = dim
        grouped["delay_rate"] = grouped["delay_rate"].round(4)
        grouped["mean_hours"] = grouped["mean_hours"].round(4)
        group_rows.append(
            grouped[
                ["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]
            ]
        )

    delay_by_group = pd.concat(group_rows, ignore_index=True)
    delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

    # Correlations matrix
    numeric_cols = model_df.select_dtypes(include=[np.number]).columns
    corr_matrix = model_df[numeric_cols].corr().round(4)
    corr_matrix.to_csv("outputs/part_a_correlations.csv", index=True, index_label="feature")

    print("Part A complete. Written modelling_table.csv and outputs/part_a_*.csv")


if __name__ == "__main__":
    run_part_a()
