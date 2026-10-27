import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
  os.makedirs("outputs", exist_ok=True)

  # Read modelling table
  df = pd.read_csv("data/modelling_table.csv")
  hubs_df = pd.read_csv("data/hubs.csv")

  # 1. Hub Profile
  # Group by hub_id: shipments (count), mean_delivery_hours, delay_rate, mean_distance_km, mean_n_stops
  grouped = (
      df.groupby("hub_id")
      .agg(
          shipments=("shipment_id", "count"),
          mean_delivery_hours=("delivery_hours", "mean"),
          delay_rate=("is_delayed", "mean"),
          mean_distance_km=("distance_km", "mean"),
          mean_n_stops=("n_stops", "mean"),
      )
      .reset_index()
  )

  # Sort by ascending hub_id (HUB01 first, HUB24 last)
  grouped = grouped.sort_values("hub_id").reset_index(drop=True)
  grouped.to_csv("outputs/hub_profile.csv", index=False)

  # 2. K-Means Segmentation & Separation Evidence
  profile_cols = [
      "mean_delivery_hours",
      "delay_rate",
      "mean_distance_km",
      "mean_n_stops",
  ]
  X_profile = grouped[profile_cols].values

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_profile)

  separation_rows = []
  for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Calculate across-to-within distance ratio
    # 276 pairs for 24 hubs (N*(N-1)/2 = 24*23/2 = 276)
    within_dists = []
    across_dists = []

    n_hubs = len(X_scaled)
    for i in range(n_hubs):
      for j in range(i + 1, n_hubs):
        dist = np.linalg.norm(X_scaled[i] - X_scaled[j])
        if labels[i] == labels[j]:
          within_dists.append(dist)
        else:
          across_dists.append(dist)

    mean_across = np.mean(across_dists) if across_dists else 0.0
    mean_within = np.mean(within_dists) if within_dists else 1.0
    ratio = mean_across / mean_within if mean_within > 0 else 0.0

    # Cluster sizes in descending order joined by hyphens
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_counts = sorted(counts, reverse=True)
    cluster_sizes_str = "-".join(map(str, sorted_counts))

    separation_rows.append({
        "K": k,
        "across_over_within": round(ratio, 3),
        "cluster_sizes": cluster_sizes_str,
    })

  pd.DataFrame(separation_rows).to_csv(
      "outputs/part_d_separation.csv", index=False
  )

  # 3. K = 3 Segment naming and tier check
  kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
  labels_3 = kmeans_3.fit_predict(X_scaled)
  grouped["cluster"] = labels_3

  # Order the three clusters by the mean of their hubs' delay_rate
  cluster_delay_means = grouped.groupby("cluster")["delay_rate"].mean()
  sorted_clusters = cluster_delay_means.sort_values().index.tolist()

  # Map cluster IDs to business names: Standard (lowest), Watch (middle), Escalate (highest)
  name_mapping = {
      sorted_clusters[0]: "Standard",
      sorted_clusters[1]: "Watch",
      sorted_clusters[2]: "Escalate",
  }
  grouped["segment"] = grouped["cluster"].map(name_mapping)

  # Merge with hubs_df to get hub_city and hub_tier
  hub_merged = grouped.merge(
      hubs_df[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left"
  )

  hub_segments = hub_merged[["hub_id", "segment", "hub_city", "hub_tier"]]
  hub_segments.to_csv("outputs/hub_segments.csv", index=False)

  # Segment profile
  segment_profile_rows = []
  for seg_name in ["Standard", "Watch", "Escalate"]:
    sub = hub_merged[hub_merged["segment"] == seg_name]
    segment_profile_rows.append({
        "segment": seg_name,
        "hubs": len(sub),
        "mean_delivery_hours": round(sub["mean_delivery_hours"].mean(), 2),
        "delay_rate": round(sub["delay_rate"].mean(), 4),
        "mean_distance_km": round(sub["mean_distance_km"].mean(), 2),
        "mean_n_stops": round(sub["mean_n_stops"].mean(), 2),
    })

  pd.DataFrame(segment_profile_rows).to_csv(
      "outputs/segment_profile.csv", index=False
  )

  # Tier vs segment mismatch check
  # Implied by tier: Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
  tier_mapping = {1: "Standard", 2: "Watch", 3: "Escalate"}
  hub_merged["implied_segment"] = hub_merged["hub_tier"].map(tier_mapping)

  mismatches = hub_merged[
      hub_merged["segment"] != hub_merged["implied_segment"]
  ]
  mismatch_out = mismatches[["hub_id", "segment", "hub_city", "hub_tier"]]
  mismatch_out.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

  print("Part D Hub profiling and segmentation complete.")


if __name__ == "__main__":
  main()