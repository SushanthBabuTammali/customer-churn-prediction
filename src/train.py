"""Customer churn prediction for a telecoms provider.

Steps
1. Load and clean the IBM Telco Customer Churn data (7,043 customers).
2. Explore churn drivers (contract type, tenure, payment method ...).
3. Train two models with the same preprocessing pipeline:
     - Logistic Regression  (simple, explainable baseline)
     - Gradient Boosting    (captures non-linear patterns)
   compared with 5-fold cross-validated ROC-AUC, then scored on a held-out test set.
4. Pick the decision threshold that minimises business cost, not the default 0.5.
5. Explain the model: which features push churn risk up or down.

Usage:  python src/train.py
"""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telco_churn.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

# Business assumptions (stated openly so they can be challenged and changed):
OFFER_COST = 100        # $ discount / call-centre cost per customer contacted
OFFER_SUCCESS = 0.30    # share of would-be churners an offer actually keeps
VALUE_MONTHS = 12       # a saved customer is worth 12 months of their own monthly bill


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    # TotalCharges is text with 11 blanks: brand-new customers (tenure 0) not yet billed.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    # Simple engineered feature: how many add-on services the customer has.
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies"]
    df["num_addons"] = (df[addons] == "Yes").sum(axis=1)
    return df.drop(columns=["customerID"])


def eda(df: pd.DataFrame) -> dict:
    plt.style.use("seaborn-v0_8-whitegrid")
    rates = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["Contract", "PaymentMethod", "InternetService"]):
        r = df.groupby(col)["Churn"].mean().sort_values() * 100
        rates[col] = r.round(1).to_dict()
        ax.barh(r.index, r.values, color="#2563eb")
        ax.set_title(f"Churn rate by {col} (%)")
        for i, v in enumerate(r.values):
            ax.text(v + 0.5, i, f"{v:.0f}%", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "churn_drivers.png", dpi=150)
    plt.close(fig)

    bins = [-1, 6, 12, 24, 48, 72]
    labels = ["0-6", "7-12", "13-24", "25-48", "49-72"]
    t = df.groupby(pd.cut(df["tenure"], bins, labels=labels), observed=True)["Churn"].mean() * 100
    rates["tenure_months"] = t.round(1).to_dict()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t.index.astype(str), t.values, color="#2563eb")
    ax.set_title("Churn rate by tenure (months) — risk is highest early on")
    ax.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(OUT / "churn_by_tenure.png", dpi=150)
    plt.close(fig)
    return rates


def build_pipelines(X: pd.DataFrame) -> dict:
    numeric = ["tenure", "MonthlyCharges", "TotalCharges", "num_addons"]
    categorical = [c for c in X.columns if c not in numeric]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
    ])
    return {
        "Logistic Regression": Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000, C=0.5))]),
        "Gradient Boosting": Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_depth=4, l2_regularization=1.0, random_state=SEED))]),
    }


def best_threshold(y_true, proba, monthly) -> tuple[float, list]:
    """Net value of a retention campaign at each threshold, versus doing nothing.

    value = (churners contacted x success rate x their 12-month value) - (customers contacted x offer cost)
    """
    y_true, monthly = np.asarray(y_true), np.asarray(monthly)
    rows = []
    for t in np.arange(0.05, 0.95, 0.01):
        contact = proba >= t
        tp = int((contact & (y_true == 1)).sum())
        fp = int((contact & (y_true == 0)).sum())
        saved_value = (monthly[contact & (y_true == 1)] * VALUE_MONTHS * OFFER_SUCCESS).sum()
        net = saved_value - contact.sum() * OFFER_COST
        rows.append({"threshold": round(float(t), 2), "net_value": int(net), "tp": tp, "fp": fp,
                     "contacted": int(contact.sum())})
    best = max(rows, key=lambda r: r["net_value"])
    return best["threshold"], rows


def main() -> None:
    df = load()
    rates = eda(df)
    X, y = df.drop(columns="Churn"), df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, pipe in build_pipelines(X).items():
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 3),
            "cv_roc_auc_std": round(float(cv_auc.std()), 3),
            "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
            "test_pr_auc": round(float(average_precision_score(y_test, proba)), 3),
        }
        fitted[name] = (pipe, proba)
        print(name, results[name])

    # Choose the model with the best cross-validated AUC (never pick on the test set).
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    pipe, proba = fitted[best_name]

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (_, p) in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {results[name]['test_roc_auc']})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Business threshold
    thr, curve = best_threshold(y_test, proba, X_test["MonthlyCharges"])
    default = next(r for r in curve if r["threshold"] == 0.5)
    chosen = next(r for r in curve if r["threshold"] == thr)
    contact_all = (np.asarray(X_test["MonthlyCharges"])[np.asarray(y_test) == 1].sum()
                   * VALUE_MONTHS * OFFER_SUCCESS - len(y_test) * OFFER_COST)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["threshold"] for r in curve], [r["net_value"] / 1000 for r in curve], color="#2563eb")
    ax.axvline(thr, ls="--", color="#dc2626", label=f"best threshold {thr}")
    ax.axvline(0.5, ls=":", color="grey", label="default 0.5")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Net campaign value ($k, test set)")
    ax.set_title("Choosing the threshold by business value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "threshold_cost.png", dpi=150)
    plt.close(fig)

    pred = (proba >= thr).astype(int)
    report = classification_report(y_test, pred, target_names=["Stay", "Churn"], output_dict=True)

    # Explainability: permutation importance on the test set (model-agnostic).
    imp = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED)
    importance = (pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False).head(8))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    importance.sort_values().plot.barh(ax=ax, color="#2563eb")
    ax.set_title(f"What drives churn predictions ({best_name})")
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Direction of effect from the logistic regression coefficients (easy to explain to a manager).
    lr = fitted["Logistic Regression"][0]
    names = lr.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(lr.named_steps["model"].coef_[0], index=names).sort_values()
    odds = {"raise_churn": (np.exp(coefs.tail(5)).round(2)).to_dict(),
            "lower_churn": (np.exp(coefs.head(5)).round(2)).to_dict()}

    summary = {
        "rows": int(len(df)), "churn_rate_pct": round(float(y.mean() * 100), 1),
        "eda_churn_rates_pct": {k: {str(a): b for a, b in v.items()} for k, v in rates.items()},
        "models": results, "chosen_model": best_name,
        "business_assumptions": {"offer_cost": OFFER_COST, "offer_success": OFFER_SUCCESS,
                                 "value_months": VALUE_MONTHS},
        "threshold": {"chosen": thr, "net_value_at_chosen": chosen["net_value"],
                      "net_value_at_0.5": default["net_value"],
                      "net_value_contact_everyone": int(contact_all),
                      "churners_caught": chosen["tp"], "churners_total": int(y_test.sum()),
                      "customers_contacted": chosen["contacted"], "test_customers": int(len(y_test))},
        "test_recall_churn": round(report["Churn"]["recall"], 3),
        "test_precision_churn": round(report["Churn"]["precision"], 3),
        "top_features": importance.round(4).to_dict(),
        "logreg_odds_ratios": odds,
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
