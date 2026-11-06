"""Part D: Model Packaging and Reload Verification.

Serializes the trained production pipeline, feature order, alert threshold,
and imputation medians into models/, then reloads the artifact from disk
and verifies exact numerical reproducibility against the in-memory model.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import FEATURES, clean_shipments, build_features


def run_package_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Load modelling table and raw data to extract medians
    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # Rebuild Part C stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Read configuration from part_c_final_config.csv
    config_df = pd.read_csv("outputs/part_c_final_config.csv")
    cfg = config_df.iloc[0]

    selector_k = int(cfg["selector_k"])
    n_estimators = int(cfg["n_estimators"])
    max_depth = None if str(cfg["max_depth"]).lower() == "unlimited" else int(cfg["max_depth"])
    min_samples_leaf = int(cfg["min_samples_leaf"])
    threshold = float(cfg["threshold"])

    final_pipeline = Pipeline(
        [
            ("selector", SelectKBest(f_classif, k=selector_k)),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                ),
            ),
        ]
    )
    final_pipeline.fit(X_train, y_train)

    # Save artifacts into models/
    #  Model pickle
    joblib.dump(final_pipeline, "models/delay_model.pkl")

    #  Threshold
    pd.DataFrame([{"threshold": threshold}]).to_csv(
        "models/delay_threshold.csv", index=False
    )

    # Feature order
    pd.DataFrame({"feature": FEATURES}).to_csv(
        "models/feature_order.csv", index=False
    )

    # Impute values
    impute_df = pd.DataFrame(
        [
            {"column": "distance_km", "median": round(float(medians["distance_km"]), 4)},
            {"column": "parcel_weight_kg", "median": round(float(medians["parcel_weight_kg"]), 4)},
            {"column": "declared_value_inr", "median": round(float(medians["declared_value_inr"]), 4)},
        ]
    )
    impute_df.to_csv("models/impute_values.csv", index=False)

    # 3. Reload check
    reloaded_pipeline = joblib.load("models/delay_model.pkl")

    in_memory_probs = final_pipeline.predict_proba(X_test)[:, 1]
    reloaded_probs = reloaded_pipeline.predict_proba(X_test)[:, 1]
    max_abs_diff = float(np.max(np.abs(in_memory_probs - reloaded_probs)))

    reload_check_df = pd.DataFrame(
        [
            ("test_rows_scored", str(len(X_test))),
            ("reload_max_abs_probability_diff", f"{max_abs_diff:.1f}"),
            ("threshold", str(threshold)),
        ],
        columns=["metric", "value"],
    )
    reload_check_df.to_csv("outputs/part_d_reload_check.csv", index=False)

    print(f"Model successfully packaged. Reload diff: {max_abs_diff}")


if __name__ == "__main__":
    run_package_model()
