import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
  os.makedirs("outputs", exist_ok=True)
  os.makedirs("models", exist_ok=True)

  # 1. Read modelling table
  df = pd.read_csv("data/modelling_table.csv")

  # Define feature columns (exact 17 features in pinned order)
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

  # Filter to existing columns in case of schema alignment
  X_cols = [c for c in feature_cols if c in df.columns]
  X = df[X_cols]
  y = df["delivery_hours"]

  # 2. Split into train (70%), temp (30%) -> val (50% of temp = 15%), test (50% of temp = 15%)
  # Sizes: 840 / 180 / 180 (Total = 1200)
  X_train, X_temp, y_train, y_temp = train_test_split(
      X, y, test_size=0.3, random_state=42
  )
  X_val, X_test, y_val, y_test = train_test_split(
      X_temp, y_temp, test_size=0.5, random_state=42
  )

  train_rows = len(X_train)
  val_rows = len(X_val)
  test_rows = len(X_test)

  # Identify numeric columns for scaling
  numeric_features = X_train.select_dtypes(
      include=[np.number]
  ).columns.tolist()

  # 3. Scaling without leakage
  scaler = StandardScaler()
  X_train_num_scaled = scaler.fit_transform(X_train[numeric_features])
  X_test_num_scaled = scaler.transform(X_test[numeric_features])

  train_scaled_max_abs_mean = float(
      np.max(np.abs(np.mean(X_train_num_scaled, axis=0)))
  )
  test_scaled_max_abs_mean = float(
      np.max(np.abs(np.mean(X_test_num_scaled, axis=0)))
  )

  checks_data = [
      ("train_rows", train_rows),
      ("val_rows", val_rows),
      ("test_rows", test_rows),
      ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
      ("test_scaled_max_abs_mean", test_scaled_max_abs_mean),
  ]
  pd.DataFrame(checks_data, columns=["metric", "value"]).to_csv(
      "outputs/part_b_checks.csv", index=False
  )

  # Preprocessing pipeline for numeric features (impute + scale) and dropping/ignoring non-numeric or handling them
  # For simplicity in pipelines with mixed types:
  numeric_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="median")),
          ("scaler", StandardScaler()),
      ]
  )

  # Preprocessor that selects numeric features (or encodes categoricals if needed, but linear/trees handle numeric or we can use numeric subset)
  # Let's use numeric features for models or handle all numeric columns in X
  preprocessor = ColumnTransformer(
      transformers=[("num", numeric_transformer, numeric_features)],
      remainder="drop",
  )

  # Helper function to evaluate model metrics
  def evaluate_model(model, X_v, y_v):
    preds = model.predict(X_v)
    mae = mean_absolute_error(y_v, preds)
    mse = mean_squared_error(y_v, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_v, preds)
    return mae, mse, rmse, r2

  # 4. Baseline Model (Predict mean of y_train for every validation row)
  mean_train_val = y_train.mean()
  val_preds_baseline = np.full_like(y_val, mean_train_val, dtype=float)
  base_mae = mean_absolute_error(y_val, val_preds_baseline)
  base_mse = mean_squared_error(y_val, val_preds_baseline)
  base_rmse = np.sqrt(base_mse)
  base_r2 = r2_score(y_val, val_preds_baseline)

  # 5. Tree-depth diagnostic
  depths = [1, 2, 3, 4, 6, 8, 12, None]
  tree_depth_results = []
  best_tree = None
  best_val_rmse = float("inf")
  best_depth_val = None

  from sklearn.tree import DecisionTreeRegressor

  for d in depths:
    # Build tree pipeline on numeric features
    tree_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor(max_depth=d, random_state=42)),
        ]
    )
    tree_pipe.fit(X_train, y_train)

    train_preds = tree_pipe.predict(X_train)
    val_preds = tree_pipe.predict(X_val)

    t_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    v_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    depth_label = "unlimited" if d is None else str(d)
    tree_depth_results.append({
        "max_depth": depth_label,
        "train_rmse": t_rmse,
        "val_rmse": v_rmse,
    })

    if v_rmse < best_val_rmse:
      best_val_rmse = v_rmse
      best_tree = tree_pipe
      best_depth_val = d
    elif v_rmse == best_val_rmse and (
        (d is not None and best_depth_val is not None and d < best_depth_val)
        or (d is None and best_depth_val is not None)
    ):
      # Tie-breaker: smaller depth
      pass

  pd.DataFrame(tree_depth_results).to_csv(
      "outputs/part_b_tree_depth.csv", index=False
  )

  # 6. Four candidate models
  # Linear Regression Pipeline
  lr_pipe = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          ("model", LinearRegression()),
      ]
  )
  lr_pipe.fit(X_train, y_train)
  lr_metrics = evaluate_model(lr_pipe, X_val, y_val)

  # Decision Tree (best from diagnostic)
  dt_metrics = evaluate_model(best_tree, X_val, y_val)

  # Random Forest
  rf_pipe = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          (
              "model",
              RandomForestRegressor(n_estimators=100, random_state=42),
          ),
      ]
  )
  rf_pipe.fit(X_train, y_train)
  rf_metrics = evaluate_model(rf_pipe, X_val, y_val)

  # Gradient Boosting
  gb_pipe = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          (
              "model",
              GradientBoostingRegressor(
                  n_estimators=100, learning_rate=0.1, random_state=42
              ),
          ),
      ]
  )
  gb_pipe.fit(X_train, y_train)
  gb_metrics = evaluate_model(gb_pipe, X_val, y_val)

  metrics_rows = [
      {
          "model": "mean_baseline",
          "split": "val",
          "MAE": base_mae,
          "MSE": base_mse,
          "RMSE": base_rmse,
          "R2": base_r2,
      },
      {
          "model": "linear_regression",
          "split": "val",
          "MAE": lr_metrics[0],
          "MSE": lr_metrics[1],
          "RMSE": lr_metrics[2],
          "R2": lr_metrics[3],
      },
      {
          "model": "decision_tree",
          "split": "val",
          "MAE": dt_metrics[0],
          "MSE": dt_metrics[1],
          "RMSE": dt_metrics[2],
          "R2": dt_metrics[3],
      },
      {
          "model": "random_forest",
          "split": "val",
          "MAE": rf_metrics[0],
          "MSE": rf_metrics[1],
          "RMSE": rf_metrics[2],
          "R2": rf_metrics[3],
      },
      {
          "model": "gradient_boosting",
          "split": "val",
          "MAE": gb_metrics[0],
          "MSE": gb_metrics[1],
          "RMSE": gb_metrics[2],
          "R2": gb_metrics[3],
      },
  ]

  # 7. Learning-rate experiment
  lr_rates = [0.01, 0.1, 1.0]
  gb_lr_results = []
  for lr in lr_rates:
    g_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=100, learning_rate=lr, random_state=42
                ),
            ),
        ]
    )
    g_pipe.fit(X_train, y_train)
    _, _, v_rmse, v_r2 = evaluate_model(g_pipe, X_val, y_val)
    gb_lr_results.append(
        {"learning_rate": lr, "val_rmse": v_rmse, "val_r2": v_r2}
    )

  pd.DataFrame(gb_lr_results).to_csv(
      "outputs/part_b_gb_learning_rate.csv", index=False
  )

  # 8. Cross-validated comparison and final model selection
  # Candidates ranked by validation RMSE: Linear Regression, Gradient Boosting, Random Forest, Decision Tree
  candidates_val_rmse = {
      "linear_regression": lr_metrics[2],
      "gradient_boosting": gb_metrics[2],
      "random_forest": rf_metrics[2],
      "decision_tree": dt_metrics[2],
  }
  sorted_candidates = sorted(candidates_val_rmse.items(), key=lambda x: x[1])
  top_two = [sorted_candidates[0][0], sorted_candidates[1][0]]

  model_objects = {
      "linear_regression": lr_pipe,
      "gradient_boosting": gb_pipe,
      "random_forest": rf_pipe,
      "decision_tree": best_tree,
  }

  # Combined train-plus-validation rows (1,020 rows)
  X_train_val = pd.concat([X_train, X_val], axis=0)
  y_train_val = pd.concat([y_train, y_val], axis=0)

  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  cv_results = []
  for name in top_two:
    # Clone estimator to fit fresh on train_val
    est = clone(model_objects[name])
    scores = cross_val_score(est, X_train_val, y_train_val, cv=kf, scoring="r2")
    cv_results.append({
        "model": name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std(),
    })

  pd.DataFrame(cv_results).to_csv("outputs/part_b_cv.csv", index=False)

  # Final model: lowest validation RMSE winner
  winner_name = sorted_candidates[0][0]
  final_model = model_objects[winner_name]
  # Fit final model on train-val or evaluate on test (fitted on training rows already or train_val)
  # The instructions state: "score it once on the test rows (already fitted on the training rows)"
  f_mae, f_mse, f_rmse, f_r2 = evaluate_model(final_model, X_test, y_test)

  metrics_rows.append({
      "model": winner_name,
      "split": "test",
      "MAE": f_mae,
      "MSE": f_mse,
      "RMSE": f_rmse,
      "R2": f_r2,
  })

  pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)
  print(
      f"Part B execution complete. Winning model ({winner_name}) evaluated on"
      " test."
  )


if __name__ == "__main__":
  main()