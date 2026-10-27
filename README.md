# Delhivery-delay-intelligence
Capstone Project
# Delhivery Delay Intelligence - Part A

## Data Findings
1. **Service Type Delays:** **Express** shipments experience delays most often, with a delay rate of **0.4897** (119 delayed out of 243 shipments), compared to Standard (0.2056) and Economy (0.0857).
2. **Hub Tier Trend:** The delay rate increases significantly as hub tiers scale down, moving from **0.0400** in Tier 1 hubs up to **0.2150** in Tier 2 and **0.4400** in Tier 3.
3. **Distance Band Impact:** Delay rates scale sharply with transit distance bands, rising from **0.0219** for 0-200 km, to **0.1368** (201-400 km), **0.4364** (401-600 km), and peaking at **0.8523** for distances of 601+ km.
4. **Top Delivery Hours Correlations:** Excluding `delivery_hours` itself, the three numeric columns most strongly correlated with `delivery_hours` (ranked by absolute correlation) are:
   - `promised_hours`: **0.8370**
   - `is_delayed`: **0.7812**
   - `distance_km`: **0.4013**
5. **Log Transformation Rationale:** `declared_value_inr` was log-transformed to stabilize skewness and normalize its right-tailed distribution; after filling missing values, its mean (**1533.05**) is substantially higher than its median (**1121.50**), whereas its log-transformed counterpart exhibits a much tighter alignment between its mean (**7.028**) and median (**7.022**).


## Model results (regression)
1. **Baseline Performance:** The mean baseline achieved a validation R² of **-0.021**, indicating that simply predicting the average delivery duration performs worse than a horizontal baseline due to inherent variance across distance bands and service tiers.
2. **Validation Winner:** **Linear Regression** (and Gradient Boosting closely following) won on validation with an RMSE of **6.15** hours (R² of **0.931**).
3. **Tree-Depth Analysis (Bias vs. Variance):** 
   - Low depths (e.g., depth 1 with a train-val RMSE gap of **0.50**) exhibit **high bias** because both training and validation errors remain high.
   - Unlimited depth exhibits **high variance**, characterized by a training RMSE of **0.00** and a large validation RMSE gap of **7.79** hours.
4. **Learning Rate Experiment:** Learning rate **0.1** performed best (val R² = **0.931**). A learning rate of **0.01** was too slow (underfitting, val R² = **0.713**), while **1.0** overshot optimal convergence (val R² = **0.844**).
5. **Cross-Validation & Deployment:** Cross-validation on the 1,020 train-val rows yielded mean R² values of **0.926** and **0.935** (with std ≤ **0.03**). Because Linear Regression is computationally lightweight, fully interpretable, and matches non-linear ensemble performance on validation, it is shipped as the production model, achieving an independent test R² of **0.949**.

## Model results (classification)
1. **Blind Baseline Performance:** The blind baseline achieves an accuracy of **0.7667** by predicting all-on-time, but it is completely useless for operations because its recall for delayed parcels is **0.0**, missing 100% of delivery failures.
2. **Noise Columns & Feature Importance:** The five noise columns account for a combined feature importance share of **0.225**, and all five correctly fall outside the top 7 features ranked by F-score.
3. **SelectKBest Feature Ranking:** The $k$-experiment shows that restricting features to the top **7** parameters provides optimal regularization and generalization on validation.
4. **Tuning Evaluation:** Tuning via grid search and randomized search yielded validation F1 scores ranging from **0.775** to **0.833**, demonstrating that while hyperparameter tuning helps, feature selection and threshold tuning drive the largest operational gains.
5. **Threshold Trade-off & Test Performance:** The alert threshold was chosen at **0.5** (satisfying the recall target ≥ 0.80), yielding robust test precision (**0.689**), recall (**0.756**), and F1 (**0.721**) across the 41 delayed test parcels.

# Delhivery Delay Intelligence Pipeline

## How to run
To reproduce the pipeline from a fresh clone, run the following commands in order:
```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
