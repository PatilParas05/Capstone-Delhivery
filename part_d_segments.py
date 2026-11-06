"""Part D: Hub Operating Segmentation and Strategic Playbooks.

Profiles all 24 regional hubs, applies K-Means clustering across operational metrics,
evaluates cluster separation across cluster counts, standardizes segment naming by delay risk,
and identifies structural mismatches between static administrative tiers and actual operational reality.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform


def run_part_d_segments():
    os.makedirs("outputs", exist_ok=True)

    # Load data
    df = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # Hub Profile: Group by hub_id, sorted by ascending hub_id
    hub_profile = (
        df.groupby("hub_id")
        .agg(
            shipments=("delivery_hours", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .reset_index()
        .sort_values("hub_id")
    )

    # Round to 4 decimal places before saving
    hub_profile_out = hub_profile.copy()
    for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        hub_profile_out[col] = hub_profile_out[col].round(4)
    hub_profile_out.to_csv("outputs/hub_profile.csv", index=False)

    # Scale the 4 profile columns
    profile_features = [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops",
    ]
    # Fixed ascending order of hub_id is guaranteed by hub_profile.sort_values('hub_id')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hub_profile[profile_features])
    dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))

    # Separation Evidence across K in [2, 3, 4, 5]
    n_hubs = len(hub_profile)
    separation_records = []
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        counts = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
        cluster_sizes_str = "-".join(str(c) for c in counts)

        within_dists = []
        across_dists = []
        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                d = dist_matrix[i, j]
                if labels[i] == labels[j]:
                    within_dists.append(d)
                else:
                    across_dists.append(d)

        ratio = float(np.mean(across_dists) / np.mean(within_dists))
        separation_records.append(
            {
                "K": k,
                "across_over_within": round(ratio, 4),
                "cluster_sizes": cluster_sizes_str,
            }
        )

    separation_df = pd.DataFrame(separation_records)
    separation_df.to_csv("outputs/part_d_separation.csv", index=False)

    # 4. K = 3 Segmentation and Segment Names
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    hub_profile["cluster"] = km3.fit_predict(X_scaled)

    # Order clusters by mean delay_rate (ascending): Standard (lowest), Watch (middle), Escalate (highest)
    cluster_delay_means = (
        hub_profile.groupby("cluster")["delay_rate"].mean().sort_values().index
    )
    name_map = {
        cluster_delay_means[0]: "Standard",
        cluster_delay_means[1]: "Watch",
        cluster_delay_means[2]: "Escalate",
    }
    hub_profile["segment"] = hub_profile["cluster"].map(name_map)

    # Merge with hubs metadata to obtain city and administrative tier
    hub_segments = hub_profile.merge(
        hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id"
    )[["hub_id", "segment", "hub_city", "hub_tier"]]
    hub_segments.to_csv("outputs/hub_segments.csv", index=False)

    # Segment Profile (means of the hub-level values)
    # Order rows: Standard, Watch, Escalate
    seg_merged = hub_profile.merge(
        hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id"
    )
    segment_profile = (
        seg_merged.groupby("segment")
        .agg(
            hubs=("hub_id", "count"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
        .reindex(["Standard", "Watch", "Escalate"])
        .reset_index()
    )

    for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        segment_profile[col] = segment_profile[col].round(4)
    segment_profile.to_csv("outputs/segment_profile.csv", index=False)

    # Tier vs Segment Mismatch
    # Expected: Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    expected_map = {1: "Standard", 2: "Watch", 3: "Escalate"}
    hub_segments["expected_segment"] = hub_segments["hub_tier"].map(expected_map)
    mismatches = hub_segments[
        hub_segments["segment"] != hub_segments["expected_segment"]
    ][["hub_id", "segment", "hub_city", "hub_tier"]]
    mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

    print("Part D segmentation complete. Outputs written to outputs/hub_*.csv, outputs/segment_profile.csv, outputs/part_d_separation.csv")


if __name__ == "__main__":
    run_part_d_segments()
