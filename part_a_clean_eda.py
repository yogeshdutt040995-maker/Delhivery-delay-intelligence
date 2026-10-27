import os
import numpy as np
import pandas as pd
from src.shipment_prep import build_features, clean_shipments


def main():
  os.makedirs("outputs", exist_ok=True)
  os.makedirs("data", exist_ok=True)

  # Load raw files
  raw_shipments = pd.read_csv("data/raw_shipments.csv")
  hubs = pd.read_csv("data/hubs.csv")

  raw_rows = len(raw_shipments)

  # Track metrics before cleaning/filling
  # 1. Duplicates count
  duplicate_rows_removed = raw_shipments.duplicated().sum()
  dedup_df = raw_shipments.drop_duplicates()
  rows_after_dedup = len(dedup_df)

  # Service type spellings before strip/title
  service_type_spellings_before = (
      dedup_df["service_type"].nunique()
      if "service_type" in dedup_df.columns
      else 0
  )

  # Missing counts before filling (on dedup_df)
  missing_distance_km_before = (
      dedup_df["distance_km"].isnull().sum()
      if "distance_km" in dedup_df.columns
      else 0
  )

  # missing_parcel_weight_before does not include sentinel rows (> 100)
  if "parcel_weight_kg" in dedup_df.columns:
    non_sentinel_weights = dedup_df["parcel_weight_kg"][
        dedup_df["parcel_weight_kg"] <= 100
    ]
    missing_parcel_weight_before = non_sentinel_weights.isnull().sum()
    sentinel_weight_rows = (dedup_df["parcel_weight_kg"] > 100).sum()
  else:
    missing_parcel_weight_before = 0
    sentinel_weight_rows = 0

  missing_declared_value_before = (
      dedup_df["declared_value"].isnull().sum()
      if "declared_value" in dedup_df.columns
      else 0
  )
  missing_payment_mode_before = (
      dedup_df["payment_mode"].isnull().sum()
      if "payment_mode" in dedup_df.columns
      else 0
  )

  # Execute clean_shipments
  clean_df, medians = clean_shipments(raw_shipments)

  service_type_spellings_after = (
      clean_df["service_type"].nunique()
      if "service_type" in clean_df.columns
      else 0
  )

  median_distance_km = medians.get("distance_km", 0.0)
  median_parcel_weight_kg = medians.get("parcel_weight_kg", 0.0)
  median_declared_value_inr = medians.get("declared_value_inr", 0.0)

  # Execute build_features and hub merge
  modelling_table = build_features(clean_df, hubs)
  rows_after_hub_merge = len(modelling_table)

  # Create target: is_delayed (1 when delivery_hours > promised_hours, else 0)
  if (
      "delivery_hours" in modelling_table.columns
      and "promised_hours" in modelling_table.columns
  ):
    modelling_table["is_delayed"] = (
        modelling_table["delivery_hours"] > modelling_table["promised_hours"]
    ).astype(int)
    

  # Save modeling table
  modelling_table.to_csv("data/modelling_table.csv", index=False)

  
  # Check nulls over 17 FEATURES + delivery_hours + is_delayed
  feature_cols = [
      "shipment_id",
      "hub_id",
      "service_type",
      "payment_mode",
      "distance_km",
      "parcel_weight_kg",
      "dispatch_time",
      "declared_value_inr",
      "hub_city",
      "region",
      "hub_tier",
      "hub_daily_capacity",
      "promised_hours",
      "is_cod",
      "is_weekend",
      "dispatch_hour",
      "log_declared_value",
  ]
  #modelling_table = pd.read_csv("data/modelling_table.csv")
  check_target_cols = feature_cols + ["delivery_hours", "is_delayed"]
  existing_check_cols = [
      c for c in check_target_cols if c in modelling_table.columns
  ]
  nulls_in_features_and_target = modelling_table[existing_check_cols].isnull().sum().sum()

  delayed_count = (
      modelling_table["is_delayed"].sum()
      if "is_delayed" in modelling_table.columns
      else 0
  )
  delay_rate = (
      delayed_count / len(modelling_table) if len(modelling_table) > 0 else 0
  )

  # Declared value metrics on filled column
  declared_value_mean_inr = modelling_table["declared_value_inr"].mean()
  declared_value_median_inr = modelling_table["declared_value_inr"].median()
  log_declared_value_mean = modelling_table["log_declared_value"].mean()
  log_declared_value_median = modelling_table["log_declared_value"].median()

  # Delivery hours outliers using 25th, 75th percentiles and 1.5 * IQR rule
  dh = modelling_table["delivery_hours"]
  q1 = dh.quantile(0.25)
  q3 = dh.quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - 1.5 * iqr
  upper_bound = q3 + 1.5 * iqr
  hours_outlier_count = ((dh < lower_bound) | (dh > upper_bound)).sum()

  # Write part_a_checks.csv
  checks_data = [
      ("raw_rows", raw_rows),
      ("duplicate_rows_removed", duplicate_rows_removed),
      ("rows_after_dedup", rows_after_dedup),
      ("missing_distance_km_before", missing_distance_km_before),
      ("missing_parcel_weight_before", missing_parcel_weight_before),
      ("sentinel_weight_rows", sentinel_weight_rows),
      ("missing_declared_value_before", missing_declared_value_before),
      ("missing_payment_mode_before", missing_payment_mode_before),
      ("service_type_spellings_before", service_type_spellings_before),
      ("service_type_spellings_after", service_type_spellings_after),
      ("median_distance_km", round(median_distance_km, 2)),
      ("median_parcel_weight_kg", round(median_parcel_weight_kg, 2)),
      ("median_declared_value_inr", round(median_declared_value_inr, 2)),
      ("rows_after_hub_merge", rows_after_hub_merge),
      ("nulls_in_features_and_target", nulls_in_features_and_target),
      ("delayed_count", delayed_count),
      ("delay_rate", round(delay_rate, 4)),
      ("declared_value_mean_inr", round(declared_value_mean_inr, 2)),
      ("declared_value_median_inr", round(declared_value_median_inr, 2)),
      ("log_declared_value_mean", round(log_declared_value_mean, 3)),
      ("log_declared_value_median", round(log_declared_value_median, 3)),
      ("hours_q1 / hours_q3", f"{q1:.2f} / {q3:.2f}"),
      (
          "hours_lower_bound / hours_upper_bound",
          f"{lower_bound:.2f} / {upper_bound:.2f}",
      ),
      ("hours_outlier_count", hours_outlier_count),
  ]
  pd.DataFrame(checks_data, columns=["metric", "value"]).to_csv(
      "outputs/part_a_checks.csv", index=False
  )

  # Delay Evidence Table
  # Distance band definition: 0-200, 201-400, 401-600, 601+ (edges 0, 200, 400, 600, 2000)
  bins = [0, 200, 400, 600, 2000]
  labels = ["0-200", "201-400", "401-600", "601+"]
  modelling_table["distance_band"] = pd.cut(
      modelling_table["distance_km"],
      bins=bins,
      labels=labels,
      include_lowest=True,
      right=True,
  )

  evidence_rows = []

  #modelling_table = pd.read_csv("data/modelling_table.csv")
  # 1. service_type dimension
  for group_val, grp in modelling_table.groupby("service_type"):
    shipments_cnt = len(grp)
    # Diagnostic check
    print("Current columns in modelling_table:", modelling_table.columns.tolist())
    print("Is 'is_delayed' in columns?", "is_delayed" in modelling_table.columns)
    delayed_cnt = int(grp["is_delayed"].sum())
    d_rate = round(delayed_cnt / shipments_cnt, 4) if shipments_cnt > 0 else 0.0
    m_hours = round(grp["delivery_hours"].mean(), 2)
    evidence_rows.append({
        "dimension": "service_type",
        "group": str(group_val),
        "shipments": shipments_cnt,
        "delayed": delayed_cnt,
        "delay_rate": d_rate,
        "mean_hours": m_hours,
    })

  # 2. hub_tier dimension
  
  for group_val, grp in modelling_table.groupby("hub_tier"):
    shipments_cnt = len(grp)
    delayed_cnt = int(grp["is_delayed"].sum())
    d_rate = round(delayed_cnt / shipments_cnt, 4) if shipments_cnt > 0 else 0.0
    m_hours = round(grp["delivery_hours"].mean(), 2)
    evidence_rows.append({
        "dimension": "hub_tier",
        "group": str(group_val),
        "shipments": shipments_cnt,
        "delayed": delayed_cnt,
        "delay_rate": d_rate,
        "mean_hours": m_hours,
    })

  # 3. distance_band dimension
  for group_val, grp in modelling_table.groupby("distance_band", observed=False):
    shipments_cnt = len(grp)
    delayed_cnt = int(grp["is_delayed"].sum())
    d_rate = round(delayed_cnt / shipments_cnt, 4) if shipments_cnt > 0 else 0.0
    m_hours = round(grp["delivery_hours"].mean(), 2)
    evidence_rows.append({
        "dimension": "distance_band",
        "group": str(group_val),
        "shipments": shipments_cnt,
        "delayed": delayed_cnt,
        "delay_rate": d_rate,
        "mean_hours": m_hours,
    })

  pd.DataFrame(evidence_rows).to_csv(
      "outputs/part_a_delay_by_group.csv", index=False
  )

  # Correlations of numeric columns
  numeric_df = modelling_table.select_dtypes(include=[np.number])
  corr_matrix = numeric_df.corr()
  corr_matrix.to_csv("outputs/part_a_correlations.csv", index=True)

  print("Part A execution complete. All outputs generated successfully.")


if __name__ == "__main__":
  main()