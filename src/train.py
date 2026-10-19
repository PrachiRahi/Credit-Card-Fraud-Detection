"""Train and compare fraud models on an imbalanced dataset.

Pipeline: stratified 60/20/20 train/val/test split
  -> 6 models (baseline, class weights, SMOTE, XGBoost variants)
  -> pick best on VALIDATION PR-AUC
  -> choose decision threshold on VALIDATION using a business cost
  -> report final numbers on the untouched TEST set.
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import FEATURES, add_features, load_raw

warnings.filterwarnings("ignore")
RS = 42


def build_models(scale_pos_weight):
    """SMOTE lives INSIDE an imblearn Pipeline, so it is only ever applied to the
    training data, never to validation/test data (this avoids data leakage)."""
    def lr(**kw):
        return LogisticRegression(max_iter=1000, **kw)

    def xgb(**kw):
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="aucpr", tree_method="hist",
                             n_jobs=-1, random_state=RS, **kw)

    return {
        "LR (baseline)": Pipeline([("scale", StandardScaler()), ("clf", lr())]),
        "LR (class_weight)": Pipeline([("scale", StandardScaler()),
                                       ("clf", lr(class_weight="balanced"))]),
        "LR + SMOTE": ImbPipeline([("scale", StandardScaler()),
                                   ("smote", SMOTE(random_state=RS)),
                                   ("clf", lr())]),
        "RandomForest (balanced)": Pipeline([("clf", RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RS))]),
        "XGBoost (scale_pos_weight)": Pipeline([("clf", xgb(scale_pos_weight=scale_pos_weight))]),
        "XGBoost + SMOTE": ImbPipeline([("smote", SMOTE(random_state=RS)),
                                        ("clf", xgb())]),
    }


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def cost_at(y, p, amount, thr, review_cost):
    """Business cost: a missed fraud costs its full amount; every flagged
    legitimate transaction costs a fixed manual-review fee."""
    pred = p >= thr
    missed = ((~pred) & (y == 1))
    false_alarm = (pred & (y == 0))
    return amount[missed].sum() + false_alarm.sum() * review_cost


def choose_threshold(y, p, amount, review_cost):
    grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    costs = [cost_at(y, p, amount, t, review_cost) for t in grid]
    return float(grid[int(np.argmin(costs))]), grid, np.array(costs)


def main(args):
    out = Path(args.out)
    for d in ("models", "reports", "data"):
        (out / d).mkdir(parents=True, exist_ok=True)

    df = add_features(load_raw(args.data))
    if args.sample:  # quick run on a subset (keeps all fraud rows)
        fraud = df[df["Class"] == 1]
        legit = df[df["Class"] == 0].sample(args.sample, random_state=RS)
        df = pd.concat([fraud, legit]).sort_values("txn_id")
    print(f"Fraud rate: {df['Class'].mean():.4%} ({int(df['Class'].sum())} frauds)")

    X, y = df[FEATURES], df["Class"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RS)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RS)
    amt_val, amt_te = df.loc[X_val.index, "Amount"].values, df.loc[X_te.index, "Amount"].values
    print(f"Split sizes -> train {len(X_tr):,} | val {len(X_val):,} | test {len(X_te):,}")

    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    models = build_models(spw)
    rows, fitted, test_scores = [], {}, {}
    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_tr, y_tr)
        p_val = model.predict_proba(X_val)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
        fitted[name], test_scores[name] = model, p_te
        m05 = metrics_at(y_te, p_te, 0.5)
        rows.append({"model": name,
                     "val_pr_auc": average_precision_score(y_val, p_val),
                     "test_pr_auc": average_precision_score(y_te, p_te),
                     "test_roc_auc": roc_auc_score(y_te, p_te),
                     "test_precision@0.5": m05["precision"],
                     "test_recall@0.5": m05["recall"],
                     "test_f1@0.5": m05["f1"]})
    res = pd.DataFrame(rows).round(4)
    res.to_csv(out / "reports/model_comparison.csv", index=False)
    print("\n", res.to_string(index=False))

    # ---- pick best model on VALIDATION PR-AUC, tune threshold on VALIDATION ----
    best_name = res.sort_values("val_pr_auc", ascending=False).iloc[0]["model"]
    best = fitted[best_name]
    p_val = best.predict_proba(X_val)[:, 1]
    thr, grid, costs = choose_threshold(y_val.values, p_val, amt_val, args.review_cost)
    print(f"\nBest model: {best_name} | chosen threshold: {thr}")

    # ---- final numbers on the untouched TEST set ----
    p_te = test_scores[best_name]
    pred = (p_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    fraud_total = amt_te[y_te.values == 1].sum()
    caught = amt_te[(pred == 1) & (y_te.values == 1)].sum()
    final = {
        "best_model": best_name, "threshold": thr, "review_cost": args.review_cost,
        "test_rows": int(len(y_te)), "test_frauds": int(y_te.sum()),
        "pr_auc": float(average_precision_score(y_te, p_te)),
        "roc_auc": float(roc_auc_score(y_te, p_te)),
        **{k: float(v) for k, v in metrics_at(y_te, p_te, thr).items()},
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "fraud_amount_total": float(fraud_total), "fraud_amount_caught": float(caught),
        "cost_with_model": float(cost_at(y_te.values, p_te, amt_te, thr, args.review_cost)),
        "cost_without_model": float(fraud_total),
    }
    json.dump(final, open(out / "reports/final_metrics.json", "w"), indent=2)
    print(json.dumps(final, indent=2))

    # ---- save artifacts ----
    # Save in a portable format: XGBoost's own JSON file works across OS/versions,
    # unlike a pickle. (Only the classifier is needed at inference; SMOTE is train-only.)
    for stale in ("best_model.joblib", "xgb_model.json"):
        (out / "models" / stale).unlink(missing_ok=True)
    clf_final = best.named_steps["clf"]
    if isinstance(clf_final, XGBClassifier):
        clf_final.save_model(str(out / "models/xgb_model.json"))
        fmt = "xgboost_json"
    else:
        joblib.dump(best, out / "models/best_model.joblib")
        fmt = "joblib"
    json.dump({"model": best_name, "threshold": thr, "format": fmt},
              open(out / "models/threshold.json", "w"))

    scored = df.loc[X_te.index, ["txn_id"] + FEATURES + ["Time", "Amount", "Class"]].copy()
    scored["fraud_score"] = p_te
    scored["predicted_fraud"] = pred
    scored[["txn_id", "fraud_score", "predicted_fraud"]].assign(
        model_name=best_name, threshold=thr).to_csv(out / "reports/test_predictions.csv", index=False)
    n_legit = int((scored["Class"] == 0).sum())
    app_sample = pd.concat([scored[scored["Class"] == 1],
                            scored[scored["Class"] == 0].sample(min(1500, n_legit), random_state=RS)])
    app_sample.to_csv(out / "data/app_sample.csv", index=False)

    # ---- plots ----
    plt.figure(figsize=(7, 5))
    for name, p in test_scores.items():
        pr, rc, _ = precision_recall_curve(y_te, p)
        plt.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_te, p):.2f})")
    plt.axhline(y_te.mean(), ls="--", c="gray", label="No-skill baseline")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves (test set)")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out / "reports/pr_curves.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(grid, costs); plt.axvline(thr, ls="--", c="red", label=f"chosen = {thr}")
    plt.xlabel("Decision threshold"); plt.ylabel("Total cost (validation)")
    plt.title("Cost vs threshold"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "reports/threshold_cost.png", dpi=150); plt.close()

    plt.figure(figsize=(3.8, 3.4))
    cm = np.array([[tn, fp], [fn, tp]])
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v:,}", ha="center", va="center")
    plt.xticks([0, 1], ["Legit", "Fraud"]); plt.yticks([0, 1], ["Legit", "Fraud"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix @ {thr}")
    plt.tight_layout(); plt.savefig(out / "reports/confusion_matrix.png", dpi=150); plt.close()

    clf = best.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values().tail(15)
        plt.figure(figsize=(6, 4.5)); imp.plot.barh()
        plt.title("Top 15 feature importances"); plt.tight_layout()
        plt.savefig(out / "reports/feature_importance.png", dpi=150); plt.close()

    if args.shap and hasattr(clf, "feature_importances_"):
        import shap
        Xs = X_te.sample(min(2000, len(X_te)), random_state=RS)
        sv = shap.TreeExplainer(clf).shap_values(Xs)
        shap.summary_plot(sv, Xs, show=False)
        plt.tight_layout(); plt.savefig(out / "reports/shap_summary.png", dpi=150); plt.close()
    print("\nDone. See the reports/ and models/ folders.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--out", default=".")
    ap.add_argument("--review_cost", type=float, default=5.0,
                    help="cost of manually reviewing one flagged transaction")
    ap.add_argument("--sample", type=int, default=0,
                    help="use only N legit rows (plus all frauds) for a quick test run")
    ap.add_argument("--shap", action="store_true", help="also create a SHAP summary plot")
    main(ap.parse_args())
