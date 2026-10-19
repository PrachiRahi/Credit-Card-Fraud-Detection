-- Analytical queries (MySQL 8 / SQLite 3.25+). Run after load_to_db.py.

-- 1. Overall fraud rate and total fraud value
SELECT COUNT(*) AS total_txns,
       SUM(is_fraud) AS fraud_txns,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct,
       ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value
FROM transactions;

-- 2. Fraud rate by hour of day (when does fraud happen?)
SELECT hour_of_day,
       COUNT(*) AS txns,
       SUM(is_fraud) AS frauds,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY fraud_rate_pct DESC;

-- 3. Fraud by amount bucket
SELECT CASE WHEN amount < 10 THEN '1) <10'
            WHEN amount < 100 THEN '2) 10-100'
            WHEN amount < 500 THEN '3) 100-500'
            ELSE '4) 500+' END AS amount_bucket,
       COUNT(*) AS txns,
       SUM(is_fraud) AS frauds,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY amount_bucket
ORDER BY amount_bucket;

-- 4. Confusion matrix of the deployed model on the test set
SELECT SUM(CASE WHEN t.is_fraud = 1 AND p.predicted_fraud = 1 THEN 1 ELSE 0 END) AS true_positives,
       SUM(CASE WHEN t.is_fraud = 0 AND p.predicted_fraud = 1 THEN 1 ELSE 0 END) AS false_positives,
       SUM(CASE WHEN t.is_fraud = 1 AND p.predicted_fraud = 0 THEN 1 ELSE 0 END) AS false_negatives,
       SUM(CASE WHEN t.is_fraud = 0 AND p.predicted_fraud = 0 THEN 1 ELSE 0 END) AS true_negatives
FROM predictions p
JOIN transactions t ON t.txn_id = p.txn_id;

-- 5. Money caught vs missed by the model (CTE)
WITH outcome AS (
    SELECT t.amount,
           CASE WHEN t.is_fraud = 1 AND p.predicted_fraud = 1 THEN 'caught'
                WHEN t.is_fraud = 1 AND p.predicted_fraud = 0 THEN 'missed'
                WHEN t.is_fraud = 0 AND p.predicted_fraud = 1 THEN 'false_alarm'
           END AS result
    FROM predictions p
    JOIN transactions t ON t.txn_id = p.txn_id
)
SELECT result, COUNT(*) AS txns, ROUND(SUM(amount), 2) AS total_amount
FROM outcome
WHERE result IS NOT NULL
GROUP BY result;

-- 6. Top 10 highest-risk transactions the model did NOT flag (window function)
SELECT * FROM (
    SELECT p.txn_id, t.amount, ROUND(p.fraud_score, 4) AS fraud_score, t.is_fraud,
           RANK() OVER (ORDER BY p.fraud_score DESC) AS risk_rank
    FROM predictions p
    JOIN transactions t ON t.txn_id = p.txn_id
    WHERE p.predicted_fraud = 0
) ranked
WHERE risk_rank <= 10;

-- 7. Hours where fraud rate is above the overall average (subquery + HAVING)
SELECT hour_of_day, COUNT(*) AS txns, SUM(is_fraud) AS frauds
FROM transactions
GROUP BY hour_of_day
HAVING 1.0 * SUM(is_fraud) / COUNT(*) > (SELECT AVG(is_fraud) FROM transactions)
ORDER BY frauds DESC;

-- 8. Compare all models
SELECT model_name, test_pr_auc, test_recall_05, test_precision_05, test_f1_05
FROM model_metrics
ORDER BY test_pr_auc DESC;
