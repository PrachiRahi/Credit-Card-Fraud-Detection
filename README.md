# Credit Card Fraud Detection (imbalanced classification + SQL + demo app)

End-to-end ML project: compare imbalance-handling strategies on the Kaggle credit card fraud dataset,
tune the decision threshold with a business cost, store results in SQL (MySQL), and serve predictions in a small Streamlit app.

> The Results section below comes from one full run (seed 42). **Re-run the code yourself and confirm the numbers on your machine before quoting them anywhere.** Small differences across library versions are normal.

## Dataset
[Kaggle: Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - 284,807 transactions, 492 frauds (0.17%).
Features `V1-V28` are anonymised PCA components; `Time` and `Amount` are raw.
Get the file with `python download_data.py` (downloads a GitHub mirror into `data/`), or download `creditcard.csv` manually from Kaggle into `data/`.
After removing 1,081 exact duplicate rows: 283,726 transactions, 473 frauds (0.167%).

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python download_data.py
cd src
python train.py --data ../data/creditcard.csv --out .. --shap     # ~5-15 min on the full data
python load_to_db.py --data ../data/creditcard.csv --root ..      # SQLite by default
cd .. && streamlit run app.py
```
Quick smoke test: add `--sample 20000` to `train.py` (uses all frauds + 20k legit rows).

**MySQL instead of SQLite:**
```bash
mysql -u root -p -e "CREATE DATABASE fraud_db;"
export DB_URL="mysql+pymysql://USER:PASSWORD@localhost:3306/fraud_db"    # Windows: set DB_URL=...
python src/load_to_db.py --data data/creditcard.csv --root .
```
Then run the queries in `sql/queries.sql` in MySQL Workbench.

**Power BI:** `load_to_db.py` writes `reports/powerbi/transactions_scored.csv` and `model_metrics.csv`.
Import them and build: fraud rate by hour, amount-bucket breakdown, caught/missed/false-alarm cards, model comparison bar chart.

## Project structure
```
data/            creditcard.csv (not committed), app_sample.csv
src/features.py  loading, de-duplication, feature engineering (hour, log_amount)
src/train.py     6-model comparison, threshold tuning, plots, artifacts
src/load_to_db.py  loads results into MySQL/SQLite + Power BI CSVs
sql/             schema.sql (3 tables), queries.sql (8 analytical queries)
app.py           Streamlit demo
models/ reports/ generated outputs (model saved as portable xgb_model.json)
```

## Method
1. **Clean + features:** dropped exact duplicate rows; `hour` from `Time`; `log_amount` from `Amount`.
2. **Split:** stratified 60/20/20 train/validation/test so each split keeps the 0.17% fraud rate.
3. **Six models:** LR baseline, LR with class weights, LR + SMOTE, Random Forest (balanced), XGBoost with `scale_pos_weight`, XGBoost + SMOTE.
   SMOTE sits inside an `imblearn` Pipeline, so it only touches training data (no leakage).
4. **Metric:** PR-AUC (average precision) as the primary metric, plus precision/recall/F1 and ROC-AUC. Accuracy is deliberately not used.
5. **Model selection** on validation PR-AUC; **threshold** chosen on validation by minimising
   `sum(amount of missed frauds) + review_cost * (false alarms)`. Final numbers are reported once on the untouched test set.
6. **Explainability:** feature importances + SHAP summary.

## Results (full run, seed 42; test set = 56,746 transactions, 95 frauds)
| Model | Val PR-AUC | Test PR-AUC | Recall @0.5 | Precision @0.5 |
|---|---|---|---|---|
| LR (baseline) | 0.700 | 0.743 | 0.611 | 0.879 |
| LR (class_weight) | 0.698 | 0.728 | 0.905 | 0.054 |
| LR + SMOTE | 0.703 | 0.730 | 0.895 | 0.051 |
| RandomForest (balanced) | 0.787 | 0.840 | 0.747 | 0.899 |
| XGBoost (scale_pos_weight) | 0.798 | 0.870 | 0.821 | 0.975 |
| XGBoost + SMOTE | 0.778 | 0.858 | 0.842 | 0.800 |

**Chosen model:** XGBoost (scale_pos_weight) (best validation PR-AUC), threshold **0.26** (tuned on validation with review cost = 5).
**Test performance at that threshold:** precision 94.0%, recall 83.2%, F1 0.883, PR-AUC 0.870
(79 frauds caught, 16 missed, 5 false alarms). Fraud value caught: 10,497 of 12,973.

What the table shows (explain this in your own words):
- Plain logistic regression has high precision but misses ~40% of fraud at 0.5.
- Class weights and SMOTE with logistic regression push recall to ~0.9 but precision collapses to ~5% (about 19 false alarms per real fraud). PR-AUC does not improve.
- Tree models (Random Forest, XGBoost) do best. XGBoost with `scale_pos_weight` beat XGBoost + SMOTE here.
- The test set has only 95 frauds, so each missed or caught fraud moves recall by about 1 point. Treat differences of 1-2 points as noise.
- SQL finding: fraud rate at 2 AM is about 1.45% vs 0.17% overall (see `sql/queries.sql`, query 2).

## Limitations / next steps
- Anonymised PCA features limit business interpretation.
- Random (not time-based) split; a production system should validate on later time periods (concept drift).
- Review cost is an assumption (default 5 per flagged transaction).
- Next: time-based validation, probability calibration, drift monitoring, API deployment.

---
## Interview prep: be able to answer these in your own words
1. **Why not accuracy?** Predicting "never fraud" gives ~99.83% accuracy and catches nothing.
2. **Why PR-AUC over ROC-AUC?** ROC-AUC counts the huge true-negative pool, so it looks good even when precision is poor. PR-AUC focuses on the rare positive class.
3. **Where can SMOTE leak?** If applied before splitting, synthetic points built from test fraud rows leak into training. Here it is inside the Pipeline and used only on train data.
4. **What did SMOTE/class weights do to precision and recall, and why?** They shift the model toward flagging more cases: recall up, precision down. Look at YOUR table and explain what you see.
5. **How did you pick the threshold?** On validation data using a cost function; the test set was untouched. What happens if review_cost is 50 instead of 5?
6. **Why XGBoost (or whichever won)?** Boosting builds trees sequentially, each fixing the previous errors; strong on tabular data. Know `scale_pos_weight` = negatives / positives.
7. **What does SHAP show?** Direction and size of each feature's effect per prediction. The V features are anonymised, so explain this limitation honestly.
8. **What would you do in production?** Time-based validation, monitoring for drift, retraining, calibration, latency and review-queue capacity.
9. **Why SQL here?** Normalised 3-table schema (transactions, predictions, model_metrics); show 2 queries you wrote and what they told you.

## Resume bullets (re-run first; edit numbers to match YOUR run)
- Built a fraud detection pipeline on <N> transactions (<x>% fraud), comparing class weighting, SMOTE and XGBoost `scale_pos_weight`, as measured by PR-AUC of <value> vs <baseline> for a logistic regression baseline.
- Tuned the decision threshold on a validation set using a cost function, achieving <recall>% recall at <precision>% precision on a held-out test set.
- Stored predictions in MySQL (3 normalised tables, 8 analytical queries) and built a Streamlit scoring demo and Power BI dashboard.
