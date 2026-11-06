"""Part D: Operational Inference and Real-Time Action Routing.

Loads incoming unlabelled dispatches (new_shipments.csv), cleans and imputes
using saved training medians, generates required features, applies the packaged
classifier and optimal decision threshold, enriches with hub segments,
and assigns targeted operations actions.
"""

import os
import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


def predict_new_shipments():
    os.makedirs("outputs", exist_ok=True)

    # Load inputs and metadata
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # Load packaged artifacts
    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"]))

    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])

    hub_segments_df = pd.read_csv("outputs/hub_segments.csv")
    seg_map = dict(zip(hub_segments_df["hub_id"], hub_segments_df["segment"]))

    # Clean and build features using shared src/shipment_prep routines
    clean_df, _ = clean_shipments(new_raw, medians=medians)
    features_df = build_features(clean_df, hubs)

    # Ensure feature alignment
    X_new = features_df[feature_order]

    # Generate delay probabilities
    model = joblib.load("models/delay_model.pkl")
    delay_probs = model.predict_proba(X_new)[:, 1].round(4)
    flagged = (delay_probs >= threshold).astype(int)

    # Map hub operational segment
    segments = clean_df["hub_id"].map(seg_map)

    # Assign playbook actions
    actions = []
    for is_flagged, segment in zip(flagged, segments):
        if is_flagged == 1:
            if segment == "Escalate":
                actions.append("Escalate to hub manager")
            else:
                actions.append("Proactive customer call")
        else:
            actions.append("No action")

    # Build output dataframe in file order
    output_df = pd.DataFrame(
        {
            "shipment_id": clean_df["shipment_id"],
            "hub_id": clean_df["hub_id"],
            "segment": segments,
            "delay_probability": delay_probs,
            "flagged": flagged,
            "action": actions,
        }
    )

    output_path = "outputs/new_shipment_actions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Scored {len(output_df)} new shipments. Output written to {output_path}")


if __name__ == "__main__":
    predict_new_shipments()
