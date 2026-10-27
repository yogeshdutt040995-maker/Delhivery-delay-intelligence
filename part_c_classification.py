import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def get_confusion_metrics(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(
      y_true, y_pred, labels=[0, 1]
  ).ravel()
  acc = accuracy_score(y_true, y_pred)
  prec = precision_score(y_true, y_pred, zero_division=0)
  rec = recall_score(y_true, y_pred, zero_division=0)
  f1 = f1_score(y_true, y_pred, zero_division=0)
  return int(tn), int(fp), int(fn), int(tp), float(acc), float(prec), float(rec), float(f1)


def main():
  os.makedirs("outputs", exist_ok=True)

  # 1. Read modelling table
  df = pd.read_csv("data/modelling_table.csv")

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

  # Stratified split: test_size=0.3 then 0.5 with stratify=y
  X_train, X_temp, y_train, y_temp = train_test_split(
      X, y, test_size=0.3, random_state=42, stratify=y
  )
  X_val, X_test, y_val, y_test = train_test_split(
      X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
  )

  # Identify numeric features for scaling/imputing
  numeric_features = X_train.select_dtypes(
      include=[np.number]
  ).columns.tolist()

  numeric_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="median")),
          ("scaler", StandardScaler()),
      ]
  )

  preprocessor = ColumnTransformer(
      transformers=[("num", numeric_transformer, numeric_features)],
      remainder="drop",
  )

  # 2. Baseline and Classifiers evaluation on validation rows (default threshold 0.5)
  models_rows = []

  # Blind baseline (predicts "not delayed" / 0 for every parcel)
  y_pred_blind = np.zeros_like(y_val)
  tn, fp, fn, tp, acc, prec, rec, f1 = get_confusion_metrics(
      y_val, y_pred_blind
  )
  models_rows.append({
      "model": "blind_all_on_time",
      "TN": tn,
      "FP": fp,
      "FN": fn,
      "TP": tp,
      "accuracy": acc,
      "precision": 0.0,
      "recall": rec,
      "f1": f1,
  })

  # Logistic Regression
  lr_pipe = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          ("classifier", LogisticRegression(random_state=42)),
      ]
  )
  lr_pipe.fit(X_train, y_train)
  y_pred_lr = lr_pipe.predict(X_val)
  tn, fp, fn, tp, acc, prec, rec, f1 = get_confusion_metrics(y_val, y_pred_lr)
  models_rows.append({
      "model": "logistic_regression",
      "TN": tn,
      "FP": fp,
      "FN": fn,
      "TP": tp,
      "accuracy": acc,
      "precision": prec,
      "recall": rec,
      "f1": f1,
  })

  # Decision Tree search over depths [2, 3, 4, 5, 6, 8, 10]
  best_dt_f1 = -1
  best_dt_depth = None
  best_dt_model = None
  for d in [2, 3, 4, 5, 6, 8, 10]:
    dt_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion="gini", max_depth=d, random_state=42
                ),
            ),
        ]
    )
    dt_pipe.fit(X_train, y_train)
    y_pred_dt = dt_pipe.predict(X_val)
    f1_val = f1_score(y_val, y_pred_dt, zero_division=0)
    if f1_val > best_dt_f1 or (
        f1_val == best_dt_f1
        and (best_dt_depth is None or d < best_dt_depth)
    ):
      best_dt_f1 = f1_val
      best_dt_depth = d
      best_dt_model = dt_pipe

  y_pred_best_dt = best_dt_model.predict(X_val)
  tn, fp, fn, tp, acc, prec, rec, f1 = get_confusion_metrics(
      y_val, y_pred_best_dt
  )
  models_rows.append({
      "model": f"decision_tree_depth_{best_dt_depth}",
      "TN": tn,
      "FP": fp,
      "FN": fn,
      "TP": tp,
      "accuracy": acc,
      "precision": prec,
      "recall": rec,
      "f1": f1,
  })

  # Random Forest Default
  rf_default = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          (
              "classifier",
              RandomForestClassifier(n_estimators=100, random_state=42),
          ),
      ]
  )
  rf_default.fit(X_train, y_train)
  y_pred_rf = rf_default.predict(X_val)
  tn, fp, fn, tp, acc, prec, rec, f1 = get_confusion_metrics(y_val, y_pred_rf)
  models_rows.append({
      "model": "random_forest_default",
      "TN": tn,
      "FP": fp,
      "FN": fn,
      "TP": tp,
      "accuracy": acc,
      "precision": prec,
      "recall": rec,
      "f1": f1,
  })

  pd.DataFrame(models_rows).to_csv("outputs/part_c_models.csv", index=False)

  # 3. Feature Importance and SelectKBest analysis
  # Fit default forest on all 17 features (numeric encoded/imputed)
  X_train_num = preprocessor.fit_transform(X_train)
  rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_all.fit(X_train_num, y_train)
  importances = rf_all.feature_importances_

  selector_all = SelectKBest(score_func=f_classif, k="all")
  selector_all.fit(X_train_num, y_train)
  f_scores = selector_all.scores_

  noise_cols = [
      "label_print_batch",
      "scanner_build_no",
      "tape_roll_no",
      "sticker_serial_mod",
      "shelf_slot_no",
  ]

  feature_scores_data = []
  for idx, col in enumerate(numeric_features):
    is_noise = col in noise_cols
    feature_scores_data.append({
        "feature": col,
        "f_score": f_scores[idx],
        "rf_importance": importances[idx],
        "is_noise": is_noise,
    })

  fs_df = pd.DataFrame(feature_scores_data)
  fs_df["f_rank"] = fs_df["f_score"].rank(ascending=False, method="min").astype(int)
  fs_df = fs_df.sort_values("f_rank")
  fs_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

  # SelectK experiment for k in [5, 7, 10, 12, 17]
  selectk_results = []
  for k_val in [5, 7, 10, 12, 17]:
    k_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(f_classif, k=k_val)),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),
        ]
    )
    k_pipe.fit(X_train, y_train)
    preds_k = k_pipe.predict(X_val)
    f1_k = f1_score(y_val, preds_k, zero_division=0)
    selectk_results.append({"k": k_val, "val_f1": f1_k})

  pd.DataFrame(selectk_results).to_csv("outputs/part_c_selectk.csv", index=False)

  # 4. Tuning with k = 7
  #selector_7 = SelectKBest(f_classif, k=7)
  #X_train_num_imputed = SimpleImputer(strategy="median").fit_transform(
   #   X_train[numeric_features]
  #)
  #X_val_num_imputed = SimpleImputer(strategy="median").transform(
   #   X_val[numeric_features]
  #)

  #X_train_k7 = selector_7.fit_transform(X_train_num_imputed, y_train)
  #X_val_k7 = selector_7.transform(X_val_num_imputed)
  # 4. Tuning with k = 7
  selector_7 = SelectKBest(f_classif, k=7)

  # Fix: Properly fit the imputer on X_train and transform both train and val
  imputer = SimpleImputer(strategy="median")
  X_train_num_imputed = imputer.fit_transform(X_train[numeric_features])
  X_val_num_imputed = imputer.transform(X_val[numeric_features])

  X_train_k7 = selector_7.fit_transform(X_train_num_imputed, y_train)
  X_val_k7 = selector_7.transform(X_val_num_imputed)

  param_grid = {
      "n_estimators": [50, 100, 150],
      "max_depth": [4, 6, 8],
      "min_samples_leaf": [1, 5, 10],
  }

  rf_tune = RandomForestClassifier(random_state=42)

  grid_search = GridSearchCV(
      estimator=rf_tune, param_grid=param_grid, cv=5, scoring="accuracy"
  )
  grid_search.fit(X_train_k7, y_train)

  rand_search = RandomizedSearchCV(
      estimator=rf_tune,
      param_distributions=param_grid,
      n_iter=9,
      cv=5,
      random_state=42,
      scoring="accuracy",
  )
  rand_search.fit(X_train_k7, y_train)

  # Untuned forest with k=7 features
  rf_untuned_7 = RandomForestClassifier(
      n_estimators=100,
      max_depth=None,
      min_samples_leaf=1,
      random_state=42,
  )
  rf_untuned_7.fit(X_train_k7, y_train)

  # Evaluate validation F1 for all three
  grid_val_f1 = f1_score(
      y_val, grid_search.predict(X_val_k7), zero_division=0
  )
  rand_val_f1 = f1_score(
      y_val, rand_search.predict(X_val_k7), zero_division=0
  )
  untuned_val_f1 = f1_score(
      y_val, rf_untuned_7.predict(X_val_k7), zero_division=0
  )

  tuning_data = [
      {
          "search": "grid_search",
          "candidates": 27,
          "cv_folds": 5,
          "total_fits": 135,
          "best_params": str(grid_search.best_params_),
          "best_cv_accuracy": grid_search.best_score_,
          "val_f1": grid_val_f1,
      },
      {
          "search": "randomized_search",
          "candidates": 9,
          "cv_folds": 5,
          "total_fits": 45,
          "best_params": str(rand_search.best_params_),
          "best_cv_accuracy": rand_search.best_score_,
          "val_f1": rand_val_f1,
      },
      {
          "search": "default_forest_same_7_features",
          "candidates": 1,
          "cv_folds": 0,
          "total_fits": 1,
          "best_params": "defaults",
          "best_cv_accuracy": np.nan,
          "val_f1": untuned_val_f1,
      },
  ]
  pd.DataFrame(tuning_data).to_csv("outputs/part_c_tuning.csv", index=False)

  # 5. Choosing final configuration
  # Ties -> default_forest_same_7_features, then grid_search, then randomized_search
  # Let's select based on max val_f1 with rule precedence:
  # Check in order: default_forest, grid_search, randomized_search or compare scores
  candidates_eval = [
      ("default_forest_same_7_features", untuned_val_f1, rf_untuned_7.get_params()),
      ("grid_search", grid_val_f1, grid_search.best_params_),
      ("randomized_search", rand_val_f1, rand_search.best_params_),
  ]
  # Find best val_f1
  best_score = max(c[1] for c in candidates_eval)
  chosen_candidate = None
  for name, score, params in candidates_eval:
    if np.isclose(score, best_score, atol=1e-4):
      chosen_candidate = name
      chosen_params = params
      break

  # Extract final hyperparameters
  if chosen_candidate == "default_forest_same_7_features":
    n_est = 100
    max_d = "unlimited"
    min_s = 1
    final_rf_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
        "random_state": 42,
    }
  elif chosen_candidate == "grid_search":
    n_est = grid_search.best_params_["n_estimators"]
    max_d = grid_search.best_params_["max_depth"]
    min_s = grid_search.best_params_["min_samples_leaf"]
    final_rf_params = {
        **grid_search.best_params_,
        "random_state": 42,
    }
  else:
    n_est = rand_search.best_params_["n_estimators"]
    max_d = rand_search.best_params_["max_depth"]
    min_s = rand_search.best_params_["min_samples_leaf"]
    final_rf_params = {
        **rand_search.best_params_,
        "random_state": 42,
    }

  final_model = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="median")),
          ("selector", SelectKBest(f_classif, k=7)),
          ("model", RandomForestClassifier(**final_rf_params)),
      ]
  )
  final_model.fit(X_train[numeric_features], y_train)

  # 6. Alert Threshold Selection
  val_probs = final_model.predict_proba(X_val[numeric_features])[:, 1]
  thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  thresh_rows = []
  chosen_threshold = 0.1

  for t in thresholds:
    y_pred_t = (val_probs >= t).astype(int)
    tn, fp, fn, tp, acc, prec, rec, f1 = get_confusion_metrics(y_val, y_pred_t)
    thresh_rows.append({
        "threshold": t,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    })
    if rec >= 0.80:
      chosen_threshold = t

  pd.DataFrame(thresh_rows).to_csv("outputs/part_c_thresholds.csv", index=False)

  # 7. Final Test Evaluation
  test_probs = final_model.predict_proba(X_test[numeric_features])[:, 1]
  y_pred_test = (test_probs >= chosen_threshold).astype(int)
  tn, fp, fn, tp, acc, prec, rec, f1 = get_confusion_metrics(
      y_pred_test, y_pred_test if False else y_test
  )

  pd.DataFrame([{
      "threshold": chosen_threshold,
      "TN": tn,
      "FP": fp,
      "FN": fn,
      "TP": tp,
      "accuracy": acc,
      "precision": prec,
      "recall": rec,
      "f1": f1,
  }]).to_csv("outputs/part_c_final_test.csv", index=False)

  pd.DataFrame([{
      "selector_k": 7,
      "n_estimators": n_est,
      "max_depth": max_d,
      "min_samples_leaf": min_s,
      "threshold": chosen_threshold,
      "chosen_from": chosen_candidate,
  }]).to_csv("outputs/part_c_final_config.csv", index=False)

  print("Part C execution complete. All classification outputs generated.")


if __name__ == "__main__":
  main()