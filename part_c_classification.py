"""Part C: Delay Classification and Operational Alert Threshold.

Implements stratified splitting, evaluates candidate classifiers, investigates
feature signal and noise elimination via SelectKBest, optimizes hyperparameters,
selects the cost-aligned decision threshold based on recall constraints,
and validates on unseen test data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return int(tn), int(fp), int(fn), int(tp), round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)


def run_part_c():
    os.makedirs("outputs", exist_ok=True)

    # Load data
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # Stratified split (840 / 180 / 180; 195 / 42 / 41 delayed)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Baseline and Candidate Classifiers
    # Blind baseline (all on-time)
    blind_pred = np.zeros(len(y_val), dtype=int)
    tn, fp, fn, tp, acc, prec, rec, f1 = compute_metrics(y_val, blind_pred)
    # Task explicitly asks for precision = 0 for blind_all_on_time
    model_rows = [
        {
            "model": "blind_all_on_time",
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "accuracy": acc,
            "precision": 0.0,
            "recall": rec,
            "f1": f1,
        }
    ]

    # Logistic Regression
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
    lr_pipe.fit(X_train, y_train)
    tn, fp, fn, tp, acc, prec, rec, f1 = compute_metrics(y_val, lr_pipe.predict(X_val))
    model_rows.append(
        {
            "model": "logistic_regression",
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    )

    # Decision Tree depth search over 
    best_dt_depth = None
    best_dt_f1 = -1.0
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        score = f1_score(y_val, dt.predict(X_val), zero_division=0)
        if score > best_dt_f1:
            best_dt_f1 = score
            best_dt_depth = d

    best_dt = DecisionTreeClassifier(
        criterion="gini", max_depth=best_dt_depth, random_state=42
    )
    best_dt.fit(X_train, y_train)
    tn, fp, fn, tp, acc, prec, rec, f1 = compute_metrics(y_val, best_dt.predict(X_val))
    model_rows.append(
        {
            "model": f"decision_tree_depth_{best_dt_depth}",
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    )

    # Default Random Forest
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_default.fit(X_train, y_train)
    tn, fp, fn, tp, acc, prec, rec, f1 = compute_metrics(y_val, rf_default.predict(X_val))
    model_rows.append(
        {
            "model": "random_forest_default",
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    )

    models_df = pd.DataFrame(model_rows)
    models_df.to_csv("outputs/part_c_models.csv", index=False)

    # Feature Signal and Noise Analysis
    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_
    rf_importances = rf_default.feature_importances_

    feat_scores_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "f_score": [round(s, 4) for s in f_scores],
            "rf_importance": [round(imp, 4) for imp in rf_importances],
            "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
        }
    )
    feat_scores_df["f_rank"] = feat_scores_df["f_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    feat_scores_df = feat_scores_df[
        ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
    ].sort_values("f_rank")
    feat_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

    # SelectK experiments for k in [5, 7, 10, 12, 17]
    selectk_records = []
    for k in [5, 7, 10, 12, 17]:
        k_pipe = Pipeline(
            [
                ("selector", SelectKBest(f_classif, k=k)),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )
        k_pipe.fit(X_train, y_train)
        val_k_f1 = f1_score(y_val, k_pipe.predict(X_val), zero_division=0)
        selectk_records.append({"k": k, "val_f1": round(val_k_f1, 4)})
    pd.DataFrame(selectk_records).to_csv("outputs/part_c_selectk.csv", index=False)

    # Tuning (k = 7)
    selector_k7 = SelectKBest(f_classif, k=7)
    X_train_k7 = selector_k7.fit_transform(X_train, y_train)
    X_val_k7 = selector_k7.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }

    # Grid Search (27 candidates x 5 folds = 135 fits)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid, cv=5
    )
    grid_search.fit(X_train_k7, y_train)
    grid_val_f1 = f1_score(y_val, grid_search.predict(X_val_k7), zero_division=0)

    # Randomized Search (9 candidates x 5 folds = 45 fits)
    rand_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        n_iter=9,
        cv=5,
        random_state=42,
    )
    rand_search.fit(X_train_k7, y_train)
    rand_val_f1 = f1_score(y_val, rand_search.predict(X_val_k7), zero_division=0)

    # Default forest on same 7 features (1 fit)
    default_k7 = RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=42
    )
    default_k7.fit(X_train_k7, y_train)
    default_val_f1 = f1_score(y_val, default_k7.predict(X_val_k7), zero_division=0)

    tuning_df = pd.DataFrame(
        [
            {
                "search": "grid_search",
                "candidates": 27,
                "cv_folds": 5,
                "total_fits": 135,
                "best_params": str(grid_search.best_params_),
                "best_cv_accuracy": round(grid_search.best_score_, 4),
                "val_f1": round(grid_val_f1, 4),
            },
            {
                "search": "randomized_search",
                "candidates": 9,
                "cv_folds": 5,
                "total_fits": 45,
                "best_params": str(rand_search.best_params_),
                "best_cv_accuracy": round(rand_search.best_score_, 4),
                "val_f1": round(rand_val_f1, 4),
            },
            {
                "search": "default_forest_same_7_features",
                "candidates": 1,
                "cv_folds": 0,
                "total_fits": 1,
                "best_params": "defaults",
                "best_cv_accuracy": "",
                "val_f1": round(default_val_f1, 4),
            },
        ]
    )
    tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)

    # Final Configuration Selection
    candidates_order = [
        ("default_forest_same_7_features", default_val_f1, {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}),
        ("grid_search", grid_val_f1, grid_search.best_params_),
        ("randomized_search", rand_val_f1, rand_search.best_params_),
    ]
    # Sort descending by val_f1; python's stable sort preserves priority order for ties
    candidates_order.sort(key=lambda x: x[1], reverse=True)
    winner_name, winner_f1, winner_params = candidates_order[0]

    # Build final full pipeline
    final_pipeline = Pipeline(
        [
            ("selector", SelectKBest(f_classif, k=7)),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    n_estimators=winner_params["n_estimators"],
                    max_depth=winner_params["max_depth"],
                    min_samples_leaf=winner_params["min_samples_leaf"],
                ),
            ),
        ]
    )
    final_pipeline.fit(X_train, y_train)

    # Alert Threshold Selection on Validation Set
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    thresh_rows = []
    for t in threshold_list:
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp, acc, prec, rec, f1 = compute_metrics(y_val, preds)
        thresh_rows.append(
            {
                "threshold": t,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )
    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv("outputs/part_c_thresholds.csv", index=False)

    # Operations rule: highest t whose validation recall is at least 0.80
    qualified = thresh_df[thresh_df["recall"] >= 0.80]
    if len(qualified) > 0:
        chosen_threshold = float(qualified["threshold"].max())
    else:
        chosen_threshold = 0.1

    # One Test Evaluation
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)
    tn, fp, fn, tp, acc, prec, rec, f1 = compute_metrics(y_test, test_preds)

    test_df = pd.DataFrame(
        [
            {
                "threshold": chosen_threshold,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        ]
    )
    test_df.to_csv("outputs/part_c_final_test.csv", index=False)

    # Final config output
    max_depth_str = (
        "unlimited"
        if winner_params["max_depth"] is None
        else str(winner_params["max_depth"])
    )
    config_df = pd.DataFrame(
        [
            {
                "selector_k": 7,
                "n_estimators": winner_params["n_estimators"],
                "max_depth": max_depth_str,
                "min_samples_leaf": winner_params["min_samples_leaf"],
                "threshold": chosen_threshold,
                "chosen_from": winner_name,
            }
        ]
    )
    config_df.to_csv("outputs/part_c_final_config.csv", index=False)

    print("Part C complete. Outputs written to outputs/part_c_*.csv")


if __name__ == "__main__":
    run_part_c()
