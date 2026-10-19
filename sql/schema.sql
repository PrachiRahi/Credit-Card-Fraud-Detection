-- Works on MySQL 8 and SQLite. Statements are separated by semicolons.
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS model_metrics;
DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    txn_id       INTEGER PRIMARY KEY,
    time_seconds DOUBLE NOT NULL,
    hour_of_day  INTEGER NOT NULL,
    amount       DOUBLE NOT NULL,
    is_fraud     INTEGER NOT NULL
);

CREATE TABLE predictions (
    txn_id          INTEGER PRIMARY KEY,
    fraud_score     DOUBLE NOT NULL,
    predicted_fraud INTEGER NOT NULL,
    model_name      VARCHAR(60) NOT NULL,
    threshold       DOUBLE NOT NULL,
    FOREIGN KEY (txn_id) REFERENCES transactions(txn_id)
);

CREATE TABLE model_metrics (
    model_name         VARCHAR(60) PRIMARY KEY,
    val_pr_auc         DOUBLE,
    test_pr_auc        DOUBLE,
    test_roc_auc       DOUBLE,
    test_precision_05  DOUBLE,
    test_recall_05     DOUBLE,
    test_f1_05         DOUBLE
);

CREATE INDEX idx_txn_hour  ON transactions(hour_of_day);
CREATE INDEX idx_txn_fraud ON transactions(is_fraud);
