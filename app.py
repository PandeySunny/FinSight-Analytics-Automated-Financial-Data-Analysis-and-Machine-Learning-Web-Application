import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_session import Session
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import traceback
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Configuration ---
UPLOAD_FOLDER = "uploads"
PLOTS_FOLDER = "static/plots"
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
SECRET_KEY = "finsight-secret-" + uuid.uuid4().hex[:16]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)
os.makedirs("flask_sessions_data", exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PLOTS_FOLDER"] = PLOTS_FOLDER
app.secret_key = SECRET_KEY
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 3600
app.config["SESSION_FILE_DIR"] = "flask_sessions_data"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

Session(app)
sns.set_theme(style="whitegrid", palette="muted")

# ──────────────────────────────────────────────────────────────────────────────
# FINANCIAL DOMAIN INTELLIGENCE
# ──────────────────────────────────────────────────────────────────────────────

FINANCIAL_COLUMN_MAP = {
    "amount":        ("transaction_amount", "numeric"),
    "balance":       ("account_balance",    "numeric"),
    "price":         ("price",              "numeric"),
    "close":         ("closing_price",      "numeric"),
    "open":          ("opening_price",      "numeric"),
    "high":          ("daily_high",         "numeric"),
    "low":           ("daily_low",          "numeric"),
    "volume":        ("trade_volume",       "numeric"),
    "revenue":       ("revenue",            "numeric"),
    "expense":       ("expense",            "numeric"),
    "profit":        ("profit",             "numeric"),
    "loss":          ("loss",               "numeric"),
    "income":        ("income",             "numeric"),
    "salary":        ("salary",             "numeric"),
    "credit":        ("credit",             "numeric"),
    "debit":         ("debit",              "numeric"),
    "loan":          ("loan_amount",        "numeric"),
    "interest":      ("interest_rate",      "numeric"),
    "tax":           ("tax",                "numeric"),
    "fee":           ("fee",                "numeric"),
    "cost":          ("cost",               "numeric"),
    "invest":        ("investment",         "numeric"),
    "return":        ("return",             "numeric"),
    "yield":         ("yield",              "numeric"),
    "dividend":      ("dividend",           "numeric"),
    "equity":        ("equity",             "numeric"),
    "asset":         ("asset",              "numeric"),
    "liabilit":      ("liability",          "numeric"),
    "debt":          ("debt",               "numeric"),
    "cash":          ("cash",               "numeric"),
    "withdraw":      ("withdrawal",         "numeric"),
    "deposit":       ("deposit",            "numeric"),
    "payment":       ("payment",            "numeric"),
    "transaction":   ("transaction",        "numeric"),
    "account":       ("account",            "categorical"),
    "customer":      ("customer",           "categorical"),
    "merchant":      ("merchant",           "categorical"),
    "category":      ("category",           "categorical"),
    "type":          ("type",               "categorical"),
    "status":        ("status",             "categorical"),
    "fraud":         ("fraud_label",        "label"),
    "label":         ("label",              "label"),
    "class":         ("class",              "label"),
    "flag":          ("flag",               "label"),
    "date":          ("date",               "datetime"),
    "time":          ("time",               "datetime"),
    "timestamp":     ("timestamp",          "datetime"),
    "period":        ("period",             "datetime"),
    "month":         ("month",              "datetime"),
    "year":          ("year",               "datetime"),
}


def classify_columns(df):
    """Map dataset columns to financial roles via heuristics."""
    roles = {"numeric": [], "categorical": [], "label": [], "datetime": [], "id": []}
    for col in df.columns:
        col_lower = col.lower()
        matched = False
        for keyword, (_, role) in FINANCIAL_COLUMN_MAP.items():
            if keyword in col_lower:
                if role == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                roles[role].append(col)
                matched = True
                break
        if not matched:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Likely ID if high cardinality & integer
                if df[col].nunique() > 0.9 * len(df) and pd.api.types.is_integer_dtype(df[col]):
                    roles["id"].append(col)
                else:
                    roles["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                roles["datetime"].append(col)
            else:
                roles["categorical"].append(col)
    return roles


def detect_dataset_type(df, roles):
    """
    Heuristically detect dataset domain:
    transactions | market_data | banking | credit_risk | general
    """
    cols_lower = [c.lower() for c in df.columns]
    score = {"transactions": 0, "market_data": 0, "banking": 0, "credit_risk": 0}

    txn_keys   = ["amount", "merchant", "transaction", "payment", "withdraw", "deposit"]
    mkt_keys   = ["open", "high", "low", "close", "volume", "price", "ticker", "symbol"]
    bank_keys  = ["balance", "account", "credit", "debit", "interest", "loan"]
    risk_keys  = ["fraud", "default", "risk", "credit_score", "label", "class"]

    for key in txn_keys:
        score["transactions"]  += sum(1 for c in cols_lower if key in c)
    for key in mkt_keys:
        score["market_data"]   += sum(1 for c in cols_lower if key in c)
    for key in bank_keys:
        score["banking"]       += sum(1 for c in cols_lower if key in c)
    for key in risk_keys:
        score["credit_risk"]   += sum(1 for c in cols_lower if key in c)

    best = max(score, key=score.get)
    return best if score[best] > 0 else "general"


# ──────────────────────────────────────────────────────────────────────────────
# ML PIPELINE — MODULAR & FINANCIAL-SECTOR FOCUSED
# ──────────────────────────────────────────────────────────────────────────────

def run_ml_pipeline(df, roles, dataset_type):
    """
    Comprehensive ML analysis:
      1. Fraud / Anomaly Detection  (Isolation Forest + optional supervised)
      2. Customer Segmentation       (K-Means + DBSCAN comparison)
      3. Risk Scoring                (rule-based quantile tiers)
      4. PCA visualization
      5. Market-specific: volatility, rolling stats (if market_data)
      6. Time-series decomposition   (if datetime present)
    Returns a rich results dict.
    """
    results = {
        "dataset_type": dataset_type,
        "anomaly": {},
        "segmentation": {},
        "risk": {},
        "pca": {},
        "market": {},
        "timeseries": {},
        "supervised": {},
        "features_used": [],
    }

    numeric_cols = roles["numeric"]
    if not numeric_cols:
        app.logger.warning("No numeric columns found — skipping ML pipeline.")
        return results

    # ── Prepare feature matrix ──
    X_raw = df[numeric_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    results["features_used"] = numeric_cols

    # ── 1. Anomaly Detection ──
    try:
        iso = IsolationForest(contamination=0.02, random_state=42, n_estimators=100)
        iso_labels = iso.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
        iso_scores = iso.decision_function(X_scaled)    # lower = more anomalous
        df["__anomaly__"] = (iso_labels == -1).astype(int)
        df["__anomaly_score__"] = -iso_scores           # invert so higher = more suspicious

        results["anomaly"] = {
            "labels":     iso_labels,
            "scores":     -iso_scores,
            "count":      int((iso_labels == -1).sum()),
            "rate_pct":   round(float((iso_labels == -1).mean() * 100), 2),
            "top_indices": list(np.argsort(-iso_scores)[:20]),  # top 20 suspicious rows
        }
        app.logger.info("Anomaly detection: %d flagged", results["anomaly"]["count"])
    except Exception as e:
        app.logger.error("Anomaly detection failed: %s", e)

    # ── 2. Customer / Transaction Segmentation ──
    try:
        n_clusters = min(5, max(2, len(df) // 500))  # adaptive cluster count
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        km_labels = kmeans.fit_predict(X_scaled)
        df["__segment__"] = km_labels

        # Segment profiles
        seg_profiles = []
        for seg_id in sorted(np.unique(km_labels)):
            mask = km_labels == seg_id
            profile = {"segment": int(seg_id), "size": int(mask.sum())}
            for col in numeric_cols[:8]:
                profile[col + "_mean"] = round(float(df.loc[mask, col].mean()), 4)
            # Anomaly rate per segment
            if "__anomaly__" in df.columns:
                profile["anomaly_rate_pct"] = round(float(df.loc[mask, "__anomaly__"].mean() * 100), 2)
            seg_profiles.append(profile)

        results["segmentation"] = {
            "labels":    km_labels,
            "n_clusters": n_clusters,
            "profiles":  seg_profiles,
            "inertia":   round(float(kmeans.inertia_), 2),
        }
        app.logger.info("Segmentation: %d clusters", n_clusters)
    except Exception as e:
        app.logger.error("Segmentation failed: %s", e)

    # ── 3. Risk Scoring (quantile-based tiers) ──
    try:
        # Use anomaly score to define risk tier
        if "scores" in results["anomaly"]:
            scores = results["anomaly"]["scores"]
            risk_tiers = pd.qcut(scores, q=4, labels=["Low", "Medium", "High", "Critical"],
                                 duplicates="drop")
            df["__risk_tier__"] = risk_tiers.astype(str)
            tier_counts = df["__risk_tier__"].value_counts().to_dict()
            results["risk"] = {
                "tier_counts": tier_counts,
                "tiers_series": risk_tiers,
            }
        app.logger.info("Risk scoring complete")
    except Exception as e:
        app.logger.error("Risk scoring failed: %s", e)

    # ── 4. PCA (2D + 3D components) ──
    try:
        n_components = min(3, X_scaled.shape[1])
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(X_scaled)
        results["pca"] = {
            "coords":           coords,
            "explained_var":    [round(v * 100, 2) for v in pca.explained_variance_ratio_],
            "cumulative_var":   round(float(pca.explained_variance_ratio_.sum() * 100), 2),
        }
        app.logger.info("PCA: %.1f%% variance explained", results["pca"]["cumulative_var"])
    except Exception as e:
        app.logger.error("PCA failed: %s", e)

    # ── 5. Market-specific analysis ──
    if dataset_type == "market_data":
        try:
            price_cols = [c for c in numeric_cols if any(k in c.lower()
                          for k in ["close", "price", "last"])]
            if price_cols:
                col = price_cols[0]
                df["__return__"]     = df[col].pct_change()
                df["__volatility__"] = df["__return__"].rolling(window=20).std() * np.sqrt(252)
                df["__ma20__"]       = df[col].rolling(20).mean()
                df["__ma50__"]       = df[col].rolling(50).mean()
                df["__rsi__"]        = compute_rsi(df[col])
                results["market"] = {
                    "price_col":   col,
                    "avg_return":  round(float(df["__return__"].mean() * 100), 4),
                    "volatility":  round(float(df["__volatility__"].mean()), 4),
                    "sharpe":      compute_sharpe(df["__return__"]),
                    "max_drawdown": compute_max_drawdown(df[col]),
                    "current_rsi": round(float(df["__rsi__"].dropna().iloc[-1]), 2)
                                   if not df["__rsi__"].dropna().empty else None,
                }
            app.logger.info("Market analysis complete")
        except Exception as e:
            app.logger.error("Market analysis failed: %s", e)

    # ── 6. Supervised fraud detection (if label column exists) ──
    label_cols = roles.get("label", [])
    if label_cols:
        try:
            label_col = label_cols[0]
            y = df[label_col].copy()
            # Encode if needed
            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y  = le.fit_transform(y.astype(str))
            y = pd.Series(y).fillna(0).astype(int)

            # Only run if at least 2 classes present
            if y.nunique() >= 2:
                X_sup = X_scaled
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sup, y, test_size=0.25, random_state=42, stratify=y)
                clf = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, random_state=42)
                clf.fit(X_tr, y_tr)
                y_pred     = clf.predict(X_te)
                y_prob     = clf.predict_proba(X_te)[:, 1]
                roc_auc    = round(float(roc_auc_score(y_te, y_prob)), 4)
                importances = dict(zip(numeric_cols,
                                       [round(float(v), 4) for v in clf.feature_importances_]))
                # sort by importance
                importances = dict(sorted(importances.items(),
                                          key=lambda x: x[1], reverse=True))
                results["supervised"] = {
                    "label_col":   label_col,
                    "roc_auc":     roc_auc,
                    "importances": importances,
                    "report":      classification_report(y_te, y_pred, output_dict=True),
                }
                app.logger.info("Supervised model ROC-AUC: %.4f", roc_auc)
        except Exception as e:
            app.logger.error("Supervised model failed: %s", e)

    return results


# ── Financial metric helpers ──

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_sharpe(returns, risk_free=0.0, periods=252):
    excess = returns - risk_free / periods
    if excess.std() == 0:
        return 0.0
    return round(float(np.sqrt(periods) * excess.mean() / excess.std()), 4)


def compute_max_drawdown(prices):
    peak = prices.cummax()
    dd   = (prices - peak) / peak
    return round(float(dd.min() * 100), 2)


# ──────────────────────────────────────────────────────────────────────────────
# FINANCIAL KPIs
# ──────────────────────────────────────────────────────────────────────────────

def compute_financial_kpis(df, roles, ml_results):
    kpis = {
        "total_records":      len(df),
        "fraud_count":        ml_results["anomaly"].get("count", 0),
        "fraud_rate_pct":     ml_results["anomaly"].get("rate_pct", 0.0),
        "num_segments":       ml_results["segmentation"].get("n_clusters", 0),
        "segment_profiles":   ml_results["segmentation"].get("profiles", []),
        "dataset_type":       ml_results.get("dataset_type", "general"),
        "pca_variance":       ml_results["pca"].get("cumulative_var", 0),
        "supervised_auc":     ml_results["supervised"].get("roc_auc"),
        "top_features":       list(ml_results["supervised"].get("importances", {}).keys())[:5],
        "risk_tier_counts":   ml_results["risk"].get("tier_counts", {}),
        "market_metrics":     ml_results.get("market", {}),
        # volume & avg from primary financial column
        "primary_col":        None,
        "total_volume":       None,
        "avg_transaction":    None,
        "median_transaction": None,
        "std_transaction":    None,
        "p95_transaction":    None,
    }

    if roles["numeric"]:
        # Prefer amount/balance/price columns
        fin_priority = ["amount", "balance", "price", "close", "revenue",
                        "payment", "transaction", "cost", "income"]
        primary = None
        for keyword in fin_priority:
            for col in roles["numeric"]:
                if keyword in col.lower():
                    primary = col
                    break
            if primary:
                break
        if not primary:
            primary = roles["numeric"][0]

        kpis["primary_col"] = primary
        series = df[primary].dropna()
        kpis["total_volume"]       = round(float(series.sum()), 2)
        kpis["avg_transaction"]    = round(float(series.mean()), 2)
        kpis["median_transaction"] = round(float(series.median()), 2)
        kpis["std_transaction"]    = round(float(series.std()), 2)
        kpis["p95_transaction"]    = round(float(series.quantile(0.95)), 2)

    return kpis


# ──────────────────────────────────────────────────────────────────────────────
# PLOT GENERATION — FINANCIAL-SECTOR CHARTS
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "primary":    "#1A56DB",
    "danger":     "#E02424",
    "success":    "#057A55",
    "warning":    "#FF8C00",
    "neutral":    "#6B7280",
    "segments":   ["#1A56DB", "#057A55", "#FF8C00", "#9333EA", "#E02424"],
    "heatmap":    "RdYlGn",
}


def generate_plots(df, prefix, ml_results, roles, dataset_type):
    plot_files = []
    numeric  = roles["numeric"]
    cat_cols = roles["categorical"]
    pca_data = ml_results["pca"].get("coords")
    anomaly  = ml_results["anomaly"]
    seg      = ml_results["segmentation"]

    # ── 1. Amount / Price Distribution with KDE ──
    for col in numeric[:3]:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            data = df[col].dropna()

            axes[0].hist(data, bins=50, color=PALETTE["primary"], alpha=0.7, edgecolor="white")
            axes[0].set_title(f"Distribution — {col}", fontweight="bold")
            axes[0].set_xlabel(col)
            axes[0].set_ylabel("Frequency")

            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr    = q3 - q1
            outlier_mask = (data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)
            axes[1].boxplot(data, vert=True, patch_artist=True,
                            boxprops=dict(facecolor=PALETTE["primary"], alpha=0.5),
                            medianprops=dict(color="red", linewidth=2))
            axes[1].set_title(f"Boxplot ({outlier_mask.sum()} outliers) — {col}", fontweight="bold")
            axes[1].set_ylabel(col)

            plt.tight_layout()
            fname = f"{prefix}_dist_{col}.png".replace(" ", "_")
            plt.savefig(os.path.join(app.config["PLOTS_FOLDER"], fname), dpi=90, bbox_inches="tight")
            plt.close()
            plot_files.append({"path": f"/static/plots/{fname}", "title": f"Distribution & Outliers: {col}",
                                "category": "distribution"})
        except Exception:
            app.logger.exception("Distribution plot failed for %s", col)

    # ── 2. Correlation Heatmap ──
    if len(numeric) >= 2:
        try:
            cols_for_corr = numeric[:12]
            corr = df[cols_for_corr].corr()
            fig, ax = plt.subplots(figsize=(max(8, len(cols_for_corr)), max(6, len(cols_for_corr) - 1)))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=len(cols_for_corr) <= 8,
                        fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
            ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=14)
            plt.tight_layout()
            fname = f"{prefix}_correlation.png"
            plt.savefig(os.path.join(app.config["PLOTS_FOLDER"], fname), dpi=90, bbox_inches="tight")
            plt.close()
            plot_files.append({"path": f"/static/plots/{fname}", "title": "Correlation Matrix",
                                "category": "correlation"})
        except Exception:
            app.logger.exception("Correlation heatmap failed")

    # ── 3. Anomaly / Fraud Scatter (PCA) ──
    if pca_data is not None and "labels" in anomaly:
        try:
            fig, ax = plt.subplots(figsize=(10, 7))
            labels  = anomaly["labels"]
            colors  = np.where(labels == -1, PALETTE["danger"], PALETTE["primary"])
            scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1],
                                 c=colors, s=12, alpha=0.5, linewidths=0)
            normal_patch  = mpatches.Patch(color=PALETTE["primary"], label=f"Normal ({(labels==1).sum():,})")
            anomaly_patch = mpatches.Patch(color=PALETTE["danger"],
                                           label=f"Anomaly/Fraud ({(labels==-1).sum():,})")
            ax.legend(handles=[normal_patch, anomaly_patch], loc="upper right", fontsize=11)
            ax.set_title("Fraud & Anomaly Detection (PCA Projection)", fontweight="bold", fontsize=14)
            ax.set_xlabel(f"PC1 ({ml_results['pca']['explained_var'][0]:.1f}% variance)")
            ax.set_ylabel(f"PC2 ({ml_results['pca']['explained_var'][1]:.1f}% variance)"
                          if len(ml_results['pca']['explained_var']) > 1 else "PC2")
            plt.tight_layout()
            fname = f"{prefix}_anomaly_pca.png"
            plt.savefig(os.path.join(app.config["PLOTS_FOLDER"], fname), dpi=90, bbox_inches="tight")
            plt.close()
            plot_files.append({"path": f"/static/plots/{fname}",
                                "title": "Anomaly / Fraud Detection",
                                "category": "fraud"})
        except Exception:
            app.logger.exception("Anomaly PCA plot failed")

    # ── 4. Customer Segmentation (PCA) ──
    if pca_data is not None and "labels" in seg:
        try:
            fig, ax = plt.subplots(figsize=(10, 7))
            seg_labels = seg["labels"]
            n_segs     = seg["n_clusters"]
            for i in range(n_segs):
                mask = seg_labels == i
                color = PALETTE["segments"][i % len(PALETTE["segments"])]
                ax.scatter(pca_data[mask, 0], pca_data[mask, 1],
                           s=15, alpha=0.6, color=color, label=f"Segment {i} (n={mask.sum():,})")
            ax.legend(loc="upper right", fontsize=10)
            ax.set_title(f"Customer Segmentation — {n_segs} Clusters (K-Means + PCA)", fontweight="bold", fontsize=14)
            ax.set_xlabel(f"PC1 ({ml_results['pca']['explained_var'][0]:.1f}% variance)")
            ax.set_ylabel(f"PC2 ({ml_results['pca']['explained_var'][1]:.1f}% variance)"
                          if len(ml_results['pca']['explained_var']) > 1 else "PC2")
            plt.tight_layout()
            fname = f"{prefix}_segmentation.png"
            plt.savefig(os.path.join(app.config["PLOTS_FOLDER"], fname), dpi=90, bbox_inches="tight")
            plt.close()
            plot_files.append({"path": f"/static/plots/{fname}",
                                "title": f"Customer Segmentation ({n_segs} Clusters)",
                                "category": "segmentation"})
        except Exception:
            app.logger.exception("Segmentation PCA plot failed")

    # ── 5. Segment Profiles Heatmap ──
    if seg.get("profiles") and len(numeric) >= 2:
        try:
            profiles    = seg["profiles"]
            profile_df  = pd.DataFrame(profiles).set_index("segment")
            metric_cols = [c for c in profile_df.columns
                           if c not in ("size", "anomaly_rate_pct")][:8]
            if metric_cols:
                profile_subset = profile_df[metric_cols]
                profile_norm   = (profile_subset - profile_subset.min()) / \
                                 (profile_subset.max() - profile_subset.min() + 1e-9)
                fig, ax = plt.subplots(figsize=(max(8, len(metric_cols)), len(profiles) + 1))
                sns.heatmap(profile_norm, annot=profile_subset.values,
                            fmt=".2f", cmap=PALETTE["heatmap"], ax=ax,
                            linewidths=0.5, cbar_kws={"label": "Normalized value"})
                ax.set_title("Segment Profile Comparison (normalised)", fontweight="bold", fontsize=14)
                ax.set_xlabel("Feature")
                ax.set_ylabel("Segment")
                plt.tight_layout()
                fname = f"{prefix}_segment_profiles.png"
                plt.savefig(os.path.join(app.config["PLOTS_FOLDER"], fname), dpi=90, bbox_inches="tight")
                plt.close()
                plot_files.append({"path": f"/static/plots/{fname}",
                                    "title": "Segment Profile Heatmap",
                                    "category": "segmentation"})
        except Exception:
            app.logger.exception("Segment profile heatmap failed")

    # ── 6. Risk Tier Distribution ──
    if ml_results["risk"].get("tier_counts"):
        try:
            tier_counts = ml_results["risk"]["tier_counts"]
            tier_order  = ["Low", "Medium", "High", "Critical"]
            labels = [t for t in tier_order if t in tier_counts]
            values = [tier_counts[t] for t in labels]
            colors = [PALETTE["success"], PALETTE["warning"], PALETTE["danger"], "#7B0000"][:len(labels)]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].bar(labels, values, color=colors, edgecolor="white", width=0.6)
            axes[0].set_title("Risk Tier Distribution", fontweight="bold", fontsize=13)
            axes[0].set_ylabel("Record Count")
            for i, (l, v) in enumerate(zip(labels, values)):
                axes[0].text(i, v + max(values)*0.01, f"{v:,}", ha="center", fontsize=10)

            axes[1].pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
                        startangle=90, pctdistance=0.8,
                        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
            axes[1].set_title("Risk Tier Breakdown", fontweight="bold", fontsize=13)

            plt.tight_layout()
            fname = f"{prefix}_risk_tiers.png"
            plt.savefig(os.path.join(app.config["PLOTS_FOLDER"], fname), dpi=90, bbox_inches="tight")
            plt.close()
            plot_files.append({"path": f"/static/plots/{fname}",
                                "title": "Risk Tier Distribution",
                                "category": "risk"})
        except Exception:
            app.logger.exception("Risk tier plot failed")

    # ── 7. Feature Importance (supervised) ──
    if ml_results["supervised"].get("importances"):
        try:
            imp  = ml_results["supervised"]["importances"]
            cols = list(imp.keys())[:12]
            vals = [imp[c] for c in cols]
            fig, ax = plt.subplots(figsize=(9, max(4, len(cols) * 0.5)))
            y_pos   = range(len(cols))
            bars    = ax.barh(y_pos, vals, color=PALETTE["primary"], alpha=0.8, edgecolor="white")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cols, fontsize=11)
            ax.invert_yaxis()
            ax.set_xlabel("Feature Importance Score")
            ax.set_title(f"Feature Importance — Fraud Model  (ROC-AUC: {ml_results['supervised']['roc_auc']:.4f})",
                         fontweight="bold", fontsize=13)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f"{val:.4f}", va="center", fontsize=9)
            plt.tight_layout()
            fname = f"{prefix}_feature_importance.png"
            plt.savefig(os.path.join(app.config["PLOTS_FOLDER"], fname), dpi=90, bbox_inches="tight")
            plt.close()
            plot_files.append({"path": f"/static/plots/{fname}",
                                "title": "Feature Importance (Fraud Model)",
                                "category": "supervised"})
        except Exception:
            app.logger.exception("Feature importance plot failed")

    # ── 8. Market / Time-series charts ──
    if dataset_type == "market_data" and "__return__" in df.columns:
        _plot_market_charts(df, prefix, plot_files, ml_results)

    # ── 9. Transaction timeline (if datetime present) ──
    if roles["datetime"]:
        _plot_time_series(df, roles, numeric, prefix, plot_files)

    # ── 10. Categorical distribution ──
    for col in cat_cols[:2]:
        try:
            counts = df[col].fillna("(Missing)").value_counts().nlargest(12)
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.barplot(x=counts.values, y=counts.index, ax=ax,
                        palette="Blues_r")
            ax.set_title(f"Top Categories — {col}", fontweight="bold", fontsize=13)
            ax.set_xlabel("Count")
            plt.tight_layout()
            fname = f"{prefix}_cat_{col}.png".replace(" ", "_")
            plt.savefig(os.path.join(app.config["PLOTS_FOLDER"], fname), dpi=90, bbox_inches="tight")
            plt.close()
            plot_files.append({"path": f"/static/plots/{fname}",
                                "title": f"Category Distribution — {col}",
                                "category": "categorical"})
        except Exception:
            app.logger.exception("Categorical plot failed for %s", col)

    return plot_files


def _plot_market_charts(df, prefix, plot_files, ml_results):
    """Generate market-specific charts: price, volatility, RSI."""
    price_col = ml_results["market"].get("price_col")
    if not price_col or price_col not in df.columns:
        return
    try:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        # Price + MAs
        df[price_col].plot(ax=axes[0], color=PALETTE["primary"], linewidth=1.2, label="Price")
        if "__ma20__" in df.columns:
            df["__ma20__"].plot(ax=axes[0], color=PALETTE["warning"], linewidth=1.2,
                                linestyle="--", label="MA20")
        if "__ma50__" in df.columns:
            df["__ma50__"].plot(ax=axes[0], color=PALETTE["danger"], linewidth=1.2,
                                linestyle="--", label="MA50")
        axes[0].set_title("Price with Moving Averages", fontweight="bold")
        axes[0].legend()
        axes[0].set_ylabel("Price")
        # Volatility
        if "__volatility__" in df.columns:
            df["__volatility__"].plot(ax=axes[1], color=PALETTE["warning"], linewidth=1.0)
            axes[1].set_title("Rolling 20-Day Annualised Volatility", fontweight="bold")
            axes[1].set_ylabel("Volatility")
        # RSI
        if "__rsi__" in df.columns:
            df["__rsi__"].plot(ax=axes[2], color=PALETTE["neutral"], linewidth=1.0)
            axes[2].axhline(70, color=PALETTE["danger"],  linestyle="--", alpha=0.7, label="Overbought (70)")
            axes[2].axhline(30, color=PALETTE["success"], linestyle="--", alpha=0.7, label="Oversold (30)")
            axes[2].set_title("RSI (14)", fontweight="bold")
            axes[2].set_ylabel("RSI")
            axes[2].legend(loc="upper right")
        plt.tight_layout()
        fname = f"{prefix}_market_analysis.png"
        plt.savefig(os.path.join(app.config["PLOTS_FOLDER"], fname), dpi=90, bbox_inches="tight")
        plt.close()
        plot_files.append({"path": f"/static/plots/{fname}",
                            "title": "Market Analysis (Price, Volatility, RSI)",
                            "category": "market"})
    except Exception:
        app.logger.exception("Market chart failed")


def _plot_time_series(df, roles, numeric_cols, prefix, plot_files):
    """Transaction volume / value over time."""
    dt_col = roles["datetime"][0]
    if dt_col not in df.columns:
        return
    try:
        df_ts = df.copy()
        df_ts[dt_col] = pd.to_datetime(df_ts[dt_col], errors="coerce")
        df_ts = df_ts.dropna(subset=[dt_col]).sort_values(dt_col)
        if df_ts.empty:
            return
        if numeric_cols:
            val_col = numeric_cols[0]
            # Resample to monthly if > 90 days span
            span_days = (df_ts[dt_col].max() - df_ts[dt_col].min()).days
            freq = "ME" if span_days > 90 else ("W" if span_days > 30 else "D")
            monthly = df_ts.set_index(dt_col)[val_col].resample(freq).sum()
            fig, axes = plt.subplots(2, 1, figsize=(13, 8))
            monthly.plot(ax=axes[0], color=PALETTE["primary"], linewidth=1.5)
            axes[0].fill_between(monthly.index, monthly.values, alpha=0.15,
                                 color=PALETTE["primary"])
            axes[0].set_title(f"Transaction Volume Over Time — {val_col}", fontweight="bold", fontsize=13)
            axes[0].set_ylabel("Total Amount")
            # Rolling mean overlay
            rolling = monthly.rolling(3).mean()
            rolling.plot(ax=axes[0], color=PALETTE["warning"], linestyle="--",
                         linewidth=1.5, label="3-period MA")
            axes[0].legend()
            # Count
            count_ts = df_ts.set_index(dt_col).resample(freq).size()
            count_ts.plot(ax=axes[1], color=PALETTE["success"], linewidth=1.5)
            axes[1].set_title("Transaction Count Over Time", fontweight="bold", fontsize=13)
            axes[1].set_ylabel("Count")
            plt.tight_layout()
            fname = f"{prefix}_timeseries.png"
            plt.savefig(os.path.join(app.config["PLOTS_FOLDER"], fname), dpi=90, bbox_inches="tight")
            plt.close()
            plot_files.append({"path": f"/static/plots/{fname}",
                                "title": "Transaction Time-Series",
                                "category": "timeseries"})
    except Exception:
        app.logger.exception("Time-series plot failed")


# ──────────────────────────────────────────────────────────────────────────────
# INSIGHTS ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def generate_financial_insights(df, ml_results, kpis, roles, dataset_type):
    """
    Generate domain-specific financial insights using rule-based analysis.
    Falls back gracefully — no external APIs required.
    """
    insights = {
        "headline":        "",
        "dataset_context": "",
        "fraud_insight":   "",
        "segment_insight": "",
        "risk_insight":    "",
        "market_insight":  "",
        "supervised_insight": "",
        "recommendations": [],
        "red_flags":       [],
        "opportunities":   [],
        "data_quality":    "",
    }

    rows, cols = df.shape
    fraud_rate = kpis["fraud_rate_pct"]

    # ── Headline ──
    domain_labels = {
        "transactions": "Transaction",
        "market_data":  "Market",
        "banking":      "Banking",
        "credit_risk":  "Credit Risk",
        "general":      "Financial",
    }
    domain = domain_labels.get(dataset_type, "Financial")
    insights["headline"] = (
        f"{domain} dataset with {rows:,} records and {cols} features. "
        f"ML pipeline identified {kpis['fraud_count']:,} anomalies "
        f"({fraud_rate}% anomaly rate) across {kpis['num_segments']} behavioural segments."
    )

    # ── Dataset context ──
    completeness = round((1 - df.isnull().mean().mean()) * 100, 1)
    insights["dataset_context"] = (
        f"Data completeness: {completeness}%. "
        f"Primary financial signal: '{kpis['primary_col']}' — "
        f"Total volume {kpis['total_volume']:,.2f} | "
        f"Mean {kpis['avg_transaction']:,.2f} | "
        f"Median {kpis['median_transaction']:,.2f} | "
        f"95th-pct {kpis['p95_transaction']:,.2f}."
        if kpis["total_volume"] is not None else
        f"Data completeness: {completeness}%."
    )

    # ── Fraud & Anomaly ──
    if fraud_rate < 1:
        insights["fraud_insight"] = (
            f"Low anomaly rate ({fraud_rate}%) — dataset appears largely clean. "
            f"Isolated flag on {kpis['fraud_count']} records warrants spot-check.")
    elif fraud_rate < 5:
        insights["fraud_insight"] = (
            f"Moderate anomaly rate ({fraud_rate}%, {kpis['fraud_count']:,} records). "
            f"Recommend secondary review of flagged transactions, especially high-value outliers.")
        insights["red_flags"].append(
            f"{kpis['fraud_count']:,} anomalous transactions at {fraud_rate}% rate — review flagged segment.")
    else:
        insights["fraud_insight"] = (
            f"⚠ High anomaly rate ({fraud_rate}%). {kpis['fraud_count']:,} records are statistically "
            f"abnormal — potential systemic fraud, data entry issues, or market stress events.")
        insights["red_flags"].append(
            f"Critical: {fraud_rate}% anomaly rate exceeds acceptable threshold (5%). Immediate audit advised.")

    # ── Segmentation ──
    profiles = kpis["segment_profiles"]
    if profiles:
        largest = max(profiles, key=lambda p: p.get("size", 0))
        riskiest = max(profiles, key=lambda p: p.get("anomaly_rate_pct", 0))
        insights["segment_insight"] = (
            f"{kpis['num_segments']} behavioural segments found. "
            f"Largest segment ({largest['segment']}) contains {largest['size']:,} records "
            f"({largest['size']/rows*100:.1f}% of total). "
            f"Highest-risk segment ({riskiest['segment']}) has "
            f"{riskiest.get('anomaly_rate_pct', 0):.1f}% anomaly rate.")
        if riskiest.get("anomaly_rate_pct", 0) > 10:
            insights["red_flags"].append(
                f"Segment {riskiest['segment']} shows {riskiest['anomaly_rate_pct']:.1f}% anomaly rate — isolate for review.")
        insights["opportunities"].append(
            f"Segment {largest['segment']} is the largest customer cohort — ideal target for retention campaigns.")

    # ── Risk Tiers ──
    tier_counts = kpis["risk_tier_counts"]
    if tier_counts:
        critical = tier_counts.get("Critical", 0)
        high     = tier_counts.get("High", 0)
        insights["risk_insight"] = (
            f"Risk tier breakdown — Critical: {critical:,} | High: {high:,} | "
            f"Medium: {tier_counts.get('Medium', 0):,} | Low: {tier_counts.get('Low', 0):,}.")
        if critical > rows * 0.05:
            insights["red_flags"].append(
                f"{critical:,} records in Critical tier — concentrate compliance review here.")

    # ── Market ──
    mkt = kpis["market_metrics"]
    if mkt:
        sharpe = mkt.get("sharpe", 0)
        vol    = mkt.get("volatility", 0)
        dd     = mkt.get("max_drawdown", 0)
        rsi    = mkt.get("current_rsi")
        insights["market_insight"] = (
            f"Sharpe ratio: {sharpe:.2f} | Annualised volatility: {vol*100:.1f}% | "
            f"Max drawdown: {dd:.1f}%."
            + (f" Current RSI: {rsi:.1f} ({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'})."
               if rsi else ""))
        if sharpe < 0:
            insights["red_flags"].append(
                f"Negative Sharpe ratio ({sharpe:.2f}) — risk-adjusted returns are subpar.")
        if abs(dd) > 20:
            insights["red_flags"].append(
                f"Max drawdown of {dd:.1f}% — significant downside exposure detected.")
        if vol > 0.3:
            insights["red_flags"].append(
                f"High annualised volatility ({vol*100:.1f}%) — consider hedging strategies.")

    # ── Supervised model ──
    if ml_results["supervised"].get("roc_auc"):
        auc = ml_results["supervised"]["roc_auc"]
        top = kpis["top_features"][:3]
        insights["supervised_insight"] = (
            f"Gradient Boosting fraud classifier trained on labelled data. "
            f"ROC-AUC: {auc:.4f} ({'Excellent' if auc > 0.9 else 'Good' if auc > 0.8 else 'Moderate'}). "
            f"Top predictive features: {', '.join(top)}.")
        if auc > 0.9:
            insights["opportunities"].append(
                f"High-performance fraud model (AUC {auc:.2f}) — ready for production deployment.")

    # ── Recommendations ──
    if fraud_rate > 2:
        insights["recommendations"].append(
            "Implement real-time transaction monitoring for anomalous patterns identified in this analysis.")
    insights["recommendations"].append(
        f"Apply segment-specific thresholds — one-size-fits-all rules miss {kpis['num_segments']} distinct behavioural groups.")
    if kpis["p95_transaction"] and kpis["avg_transaction"]:
        ratio = kpis["p95_transaction"] / (kpis["avg_transaction"] or 1)
        if ratio > 5:
            insights["recommendations"].append(
                f"95th-percentile value is {ratio:.0f}x the mean — consider separate risk policies for large transactions.")
    if completeness < 90:
        insights["recommendations"].append(
            f"Data completeness at {completeness}% — missing values may bias ML model outputs; review data collection pipeline.")
    insights["recommendations"].append(
        "Schedule quarterly model retraining to capture evolving transaction patterns and adversarial drift.")

    # ── Data quality ──
    dupes = df.duplicated().sum()
    missing_pct = round(df.isnull().mean().mean() * 100, 1)
    quality_score = int(completeness - dupes / rows * 50)
    insights["data_quality"] = (
        f"Quality score: {min(100, quality_score)}/100 | "
        f"Missing: {missing_pct}% | Duplicates: {dupes:,} | "
        f"Columns: {cols} ({len(roles['numeric'])} numeric, {len(roles['categorical'])} categorical)")

    return insights


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_path(folder, name):
    base, ext = os.path.splitext(name)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    uniq  = f"{base}_{stamp}_{uuid.uuid4().hex[:6]}{ext}"
    return os.path.join(folder, uniq)


def convert_datetime_columns(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            try:
                df_copy[f"{col}_year"]  = df_copy[col].dt.year
                df_copy[f"{col}_month"] = df_copy[col].dt.month
                df_copy[f"{col}_day"]   = df_copy[col].dt.day
            except Exception:
                pass
    return df_copy


def cleanup_old_files(max_age_seconds=3600):
    import time
    for folder in [app.config["UPLOAD_FOLDER"], app.config["PLOTS_FOLDER"]]:
        if not os.path.exists(folder):
            continue
        now = time.time()
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                if os.path.isfile(fpath) and not fname.startswith("."):
                    if now - os.path.getmtime(fpath) > max_age_seconds:
                        os.remove(fpath)
            except Exception:
                pass


def dataset_summary(df):
    rows = []
    for col in df.columns:
        try:
            dtype    = str(df[col].dtype)
            non_null = int(df[col].notna().sum())
            unique   = int(df[col].nunique(dropna=True))
            sample   = str(df[col].dropna().iloc[0]) if non_null > 0 else ""
            info     = {"Column": col, "Type": dtype,
                        "Non-Null": non_null, "Unique": unique,
                        "Sample": sample[:40]}
            if pd.api.types.is_numeric_dtype(df[col]):
                info["Mean"] = round(float(df[col].mean(skipna=True)), 4) if non_null else None
                info["Std"]  = round(float(df[col].std(skipna=True)),  4) if non_null else None
            else:
                info["Mean"] = None
                info["Std"]  = None
            rows.append(info)
        except Exception:
            pass
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Please select a file to upload.")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Supported formats: CSV, XLSX, XLS")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        path     = unique_path(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)
        size_mb  = os.path.getsize(path) / (1024 * 1024)
        session["uploaded_filename"] = os.path.basename(path)
        session.permanent = True
        app.logger.info("Saved %s (%.2f MB)", path, size_mb)
        return redirect(url_for("results"))

    return render_template("upload.html",
                           uploaded_filename=session.get("uploaded_filename"))


@app.route("/results")
def results():
    try:
        basename = session.get("uploaded_filename")
        if not basename:
            flash("No file uploaded. Please upload a dataset first.")
            return redirect(url_for("upload_file"))

        full_path = os.path.join(app.config["UPLOAD_FOLDER"], basename)
        if not os.path.exists(full_path):
            flash("File missing — please re-upload.")
            session.pop("uploaded_filename", None)
            return redirect(url_for("upload_file"))

        cleanup_old_files()

        # ── Load ──
        ext = basename.rsplit(".", 1)[-1].lower()
        try:
            if ext == "csv":
                df = pd.read_csv(full_path, low_memory=False, parse_dates=True)
            elif ext in ["xlsx", "xls"]:
                df = pd.read_excel(full_path, engine="openpyxl" if ext == "xlsx" else "xlrd")
            else:
                flash(f"Unsupported file format: {ext}. Use CSV, XLSX, or XLS.")
                return redirect(url_for("upload_file"))
            app.logger.info("Loaded %s shape=%s", basename, df.shape)
        except MemoryError:
            try:
                df = pd.read_csv(full_path, nrows=100_000, parse_dates=True)
                flash("File too large for full load — analysing first 100k rows.")
            except Exception as e:
                flash(f"File too large and failed to load partial data: {str(e)[:100]}")
                return redirect(url_for("upload_file"))
        except Exception as e:
            app.logger.error("File load failed: %s", e)
            flash(f"Failed to load file: {str(e)[:150]}. Check file format and structure.")
            return redirect(url_for("upload_file"))

        if df is None or df.empty:
            flash("Dataset is empty or could not be read.")
            return redirect(url_for("upload_file"))

        # ── Convert datetime columns ──
        df = convert_datetime_columns(df)

        # ── Light cleaning (keep NaN so ML can impute; only drop all-NaN rows) ──
        df = df.dropna(how="all").drop_duplicates()

        if df.empty:
            flash("Dataset empty after cleaning.")
            return redirect(url_for("upload_file"))

        # Sample for ML / plots if huge
        if len(df) > 200_000:
            sample_df = df.sample(n=200_000, random_state=42)
        else:
            sample_df = df

        rows, cols = sample_df.shape

        # ── Column classification ──
        try:
            roles        = classify_columns(sample_df)
            dataset_type = detect_dataset_type(sample_df, roles)
            app.logger.info("Dataset type: %s", dataset_type)
        except Exception as e:
            app.logger.exception("Column classification failed: %s", e)
            flash(f"Column classification failed: {str(e)[:100]}")
            return redirect(url_for("upload_file"))

        # ── ML Pipeline ──
        try:
            app.logger.info("Running ML pipeline…")
            ml_results = run_ml_pipeline(sample_df, roles, dataset_type)
        except Exception as e:
            app.logger.exception("ML pipeline failed: %s", e)
            flash(f"ML analysis failed: {str(e)[:150]}. Try a different dataset.")
            return redirect(url_for("upload_file"))

        # ── KPIs ──
        try:
            kpis = compute_financial_kpis(sample_df, roles, ml_results)
        except Exception as e:
            app.logger.exception("KPI computation failed: %s", e)
            kpis = {"total_records": len(sample_df), "fraud_count": 0}

        # ── Insights ──
        try:
            insights = generate_financial_insights(sample_df, ml_results, kpis, roles, dataset_type)
        except Exception as e:
            app.logger.exception("Insights generation failed: %s", e)
            insights = {"headline": "Analysis complete", "recommendations": []}

        # ── Plots ──
        prefix = os.path.splitext(basename)[0]
        plots = []
        try:
            plots = generate_plots(sample_df, prefix, ml_results, roles, dataset_type)
        except Exception as e:
            app.logger.exception("Plot generation failed: %s", e)
            flash("Some visualizations could not be generated, but analysis is complete.")

        # ── Summary table ──
        summary_df   = dataset_summary(sample_df)
        summary_html = summary_df.to_html(classes="summary-table", index=False,
                                          float_format="%.4f", na_rep="—")
        head_html    = sample_df.head(15).to_html(classes="data-table", index=False,
                                                   escape=False)

        return render_template(
            "results.html",
            filename=basename,
            rows=rows,
            cols=cols,
            dataset_type=dataset_type,
            roles=roles,
            head_html=head_html,
            summary_html=summary_html,
            plots=plots,
            ml_results=ml_results,
            kpis=kpis,
            financial_kpis=kpis,
            insights=insights,
        )

    except Exception as e:
        app.logger.exception("Unexpected error in /results: %s", e)
        flash(f"Analysis failed: {e}")
        return redirect(url_for("upload_file"))


@app.errorhandler(413)
def too_large(e):
    return "File too large (max 500 MB).", 413


if __name__ == "__main__":
    app.run(debug=True, threaded=True)