"""Part B: Delivery-Hours Regression Pipeline.

Splits data into train/val/test, validates leakage-free scaling, evaluates
candidate regression algorithms, diagnoses tree depths and learning rates,
conducts cross-validation, and evaluates the final winning model on unseen test data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES


def run_part_b():
    os.makedirs("outputs", exist_ok=True)

    # Load modelling table
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]

    # Split into training, validation, and test (840 / 180 / 180)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Scaling without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = float(np.abs(X_train_scaled.mean(axis=0)).max())
    test_scaled_max_abs_mean = float(np.abs(X_test_scaled.mean(axis=0)).max())

    checks_df = pd.DataFrame(
        [
            ("train_rows", str(len(X_train))),
            ("val_rows", str(len(X_val))),
            ("test_rows", str(len(X_test))),
            ("train_scaled_max_abs_mean", f"{train_scaled_max_abs_mean:.4e}"),
            ("test_scaled_max_abs_mean", str(round(test_scaled_max_abs_mean, 4))),
        ],
        columns=["metric", "value"],
    )
    checks_df.to_csv("outputs/part_b_checks.csv", index=False)

    # Baseline: predict mean of y_train
    mean_val_pred = np.full_like(y_val, y_train.mean())
    base_mae = mean_absolute_error(y_val, mean_val_pred)
    base_mse = mean_squared_error(y_val, mean_val_pred)
    base_rmse = np.sqrt(base_mse)
    base_r2 = r2_score(y_val, mean_val_pred)

    # Tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_records = []
    best_depth = None
    lowest_tree_val_rmse = float("inf")

    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        train_pred = dt.predict(X_train)
        val_pred = dt.predict(X_val)

        tr_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        va_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        depth_label = str(d) if d is not None else "unlimited"
        tree_records.append(
            {
                "max_depth": depth_label,
                "train_rmse": round(tr_rmse, 4),
                "val_rmse": round(va_rmse, 4),
            }
        )

        if va_rmse < lowest_tree_val_rmse:
            lowest_tree_val_rmse = va_rmse
            best_depth = d

    tree_depth_df = pd.DataFrame(tree_records)
    tree_depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)

    # Four candidate models
    candidates = {
        "linear_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }

    metrics_rows = [
        {
            "model": "mean_baseline",
            "split": "val",
            "MAE": round(base_mae, 4),
            "MSE": round(base_mse, 4),
            "RMSE": round(base_rmse, 4),
            "R2": round(base_r2, 4),
        }
    ]

    val_rmses = {}
    fitted_models = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, pred)
        mse = mean_squared_error(y_val, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, pred)

        val_rmses[name] = rmse
        metrics_rows.append(
            {
                "model": name,
                "split": "val",
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
            }
        )

    # Learning-rate experiment
    lr_records = []
    for lr in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(
            n_estimators=100, learning_rate=lr, random_state=42
        )
        gb.fit(X_train, y_train)
        gb_val_pred = gb.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, gb_val_pred))
        val_r2 = r2_score(y_val, gb_val_pred)
        lr_records.append(
            {
                "learning_rate": lr,
                "val_rmse": round(val_rmse, 4),
                "val_r2": round(val_r2, 4),
            }
        )
    pd.DataFrame(lr_records).to_csv(
        "outputs/part_b_gb_learning_rate.csv", index=False
    )

    # Cross-validated comparison and final model
    # Rank candidates by validation RMSE
    ranked_candidates = sorted(val_rmses.items(), key=lambda x: x[1])
    top_2_names = [ranked_candidates[0][0], ranked_candidates[1][0]]

    # Shared KFold on combined train + val (1020 rows)
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_rows = []
    for name in top_2_names:
        model = candidates[name]
        scores = cross_val_score(model, X_train_val, y_train_val, cv=kf)
        cv_rows.append(
            {
                "model": name,
                "mean_r2": round(scores.mean(), 4),
                "std_r2": round(scores.std(), 4),
            }
        )
    pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

    # Final model is candidate with lowest validation RMSE
    final_model_name = ranked_candidates[0][0]
    final_model = fitted_models[final_model_name]

    # Evaluate final model on test rows (already fitted on training rows)
    test_pred = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_pred)

    metrics_rows.append(
        {
            "model": final_model_name,
            "split": "test",
            "MAE": round(test_mae, 4),
            "MSE": round(test_mse, 4),
            "RMSE": round(test_rmse, 4),
            "R2": round(test_r2, 4),
        }
    )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)

    print("Part B complete. Outputs written to outputs/part_b_*.csv")


if __name__ == "__main__":
    run_part_b()
