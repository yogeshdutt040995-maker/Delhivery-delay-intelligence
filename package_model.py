import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.shipment_prep import clean_shipments


def main():
  os.makedirs("models", exist_ok=True)
  os.makedirs("outputs", exist_ok=True)

  # 1. Read data and rebuild stratified split matching Part C
  df = pd.read_csv("data/modelling_table.csv")
  raw_shipments = pd.read_csv("data/raw_shipments.csv")
  _, medians = clean_shipments(raw_shipments)

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

  X_cols = [c for c in feature_cols if c in df.columns]
  X = df[X_cols]
  y = df["is_delayed"]

  X_train, X_temp, y_train, y_temp = train_test_split(
      X, y, test_size=0.3, random_state=42, stratify=y
  )
  X_val, X_test, y_val, y_test = train_test_split(
      X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
  )

  numeric_features = X_train.select_dtypes(
      include=[np.number]
  ).columns.tolist()

  # Load chosen config from Part C final config if available, or use defaults/chosen parameters
  config_path = "outputs/part_c_final_config.csv"
  if os.path.exists(config_path):
    cfg = pd.read_csv(config_path).iloc[0]
    n_est = int(cfg["n_estimators"])
    max_d = None if str(cfg["max_depth"]).lower() == "unlimited" else int(cfg["max_depth"])
    min_s = int(cfg["min_samples_leaf"])
    threshold = float(cfg["threshold"])
  else:
    n_est = 100
    max_d = None
    min_s = 1
    threshold = 0.5

  # Fit Final Pipeline
  final_pipeline = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="median")),
          ("selector", SelectKBest(f_classif, k=7)),
          (
              "model",
              RandomForestClassifier(
                  n_estimators=n_est,
                  max_depth=max_d,
                  min_samples_leaf=min_s,
                  random_state=42,
              ),
          ),
      ]
  )

  final_pipeline.fit(X_train[numeric_features], y_train)

  # Save artifacts
  joblib.dump(final_pipeline, "models/delay_model.pkl")

  pd.DataFrame({"threshold": [threshold]}).to_csv(
      "models/delay_threshold.csv", index=False
  )
  pd.DataFrame({"feature": numeric_features}).to_csv(
      "models/feature_order.csv", index=False
  )

  median_rows = [{"column": col, "median": medians[col]} for col in numeric_features if col in medians]
  pd.DataFrame(median_rows).to_csv("models/impute_values.csv", index=False)

  # Reload check
  reloaded_pipeline = joblib.load("models/delay_model.pkl")

  in_mem_probs = final_pipeline.predict_proba(X_test[numeric_features])[:, 1]
  reload_probs = reloaded_pipeline.predict_proba(X_test[numeric_features])[:, 1]
  max_diff = float(np.max(np.abs(in_mem_probs - reload_probs)))

  reload_check_data = [
      ("test_rows_scored", len(X_test)),
      ("reload_max_abs_probability_diff", max_diff),
      ("threshold", threshold),
  ]
  pd.DataFrame(reload_check_data, columns=["metric", "value"]).to_csv(
      "outputs/part_d_reload_check.csv", index=False
  )

  print(f"Model packaged successfully. Reload max abs diff: {max_diff}")


if __name__ == "__main__":
  main()