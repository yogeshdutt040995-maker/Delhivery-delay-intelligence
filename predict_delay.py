import os
import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import build_features, clean_shipments


def main():
  os.makedirs("outputs", exist_ok=True)

  # 1. Load data and models/artifacts
  new_raw = pd.read_csv("data/new_shipments.csv")
  hubs = pd.read_csv("data/hubs.csv")
  hub_segments = pd.read_csv("outputs/hub_segments.csv")

  model = joblib.load("models/delay_model.pkl")
  threshold_df = pd.read_csv("models/delay_threshold.csv")
  threshold = float(threshold_df["threshold"].iloc[0])

  feature_order_df = pd.read_csv("models/feature_order.csv")
  feature_order = feature_order_df["feature"].tolist()

  impute_df = pd.read_csv("models/impute_values.csv")
  medians = dict(zip(impute_df["column"], impute_df["median"]))

  # 2. Clean new shipments using shared clean_shipments and stored medians
  clean_new, _ = clean_shipments(new_raw, medians=medians)

  # 3. Attach features using shared build_features
  featured_new = build_features(clean_new, hubs)

  # 4. Arrange columns in feature order
  X_new = featured_new[feature_order]

  # 5. Predict probabilities
  probs = model.predict_proba(X_new)[:, 1]
  featured_new["delay_probability"] = np.round(probs, 4)

  # 6. Flagged status
  featured_new["flagged"] = (featured_new["delay_probability"] >= threshold).astype(int)

  # 7. Join hub segment
  merged_actions = featured_new.merge(
      hub_segments[["hub_id", "segment"]], on="hub_id", how="left"
  )

  # 8. Assign action rules
  # - Escalate to hub manager: flagged == 1 and segment == 'Escalate'
  # - Proactive customer call: flagged == 1 and segment != 'Escalate'
  # - No action: flagged == 0
  def assign_action(row):
    if row["flagged"] == 0:
      return "No action"
    elif row["segment"] == "Escalate":
      return "Escalate to hub manager"
    else:
      return "Proactive customer call"

  merged_actions["action"] = merged_actions.apply(assign_action, axis=1)

  # Select output columns in file order
  output_cols = [
      "shipment_id",
      "hub_id",
      "segment",
      "delay_probability",
      "flagged",
      "action",
  ]
  final_output = merged_actions[output_cols]
  final_output.to_csv("outputs/new_shipment_actions.csv", index=False)

  print("Scoring complete. Actions saved to outputs/new_shipment_actions.csv")


if __name__ == "__main__":
  main()