import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
)
import joblib
import shap

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# PROJECT ROOT
# ──────────────────────────────────────────────────────────────
def find_project_root():
    candidates = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for start in candidates:
        for p in [start, *start.parents]:
            if (p / "data").is_dir() and (p / "models_saved").is_dir():
                return p
    return Path.cwd().resolve()


ROOT = find_project_root()
print(f"📁 Project root: {ROOT}")

RAW_DATA   = ROOT / "data" / "raw" / "finguard-fraud-detection-dataset.csv"
PROCESSED  = ROOT / "data" / "processed"
FIG_DIR    = ROOT / "reports" / "figuresNewV4"
MODEL_DIR  = ROOT / "models_saved"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# ⭐ PUBLICATION-GRADE STYLE (Arial Font + Bolder Ticks)
# ──────────────────────────────────────────────────────────────
DPI = 600
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

mpl.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,

    # ⭐ Fonts: Changed to Arial + Bigger & Bolder
    "font.family": "sans-serif",              # Switched to sans-serif
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,                          
    "axes.titlesize": 16,                     
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelsize": 14,                     
    "axes.labelweight": "bold",               
    "axes.labelpad": 10,

    # Ticks
    "xtick.labelsize": 12,                    
    "ytick.labelsize": 12,                    
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Legend
    "legend.fontsize": 12,                    
    "legend.title_fontsize": 13,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "black",

    # Axes
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.4,                    
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,

    # Lines
    "lines.linewidth": 2.4,                   

    # Save
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.pad_inches": 0.15,
})

PALETTE = {
    "fraud":     "#C0392B",
    "nonfraud":  "#27AE60",
    "primary":   "#2C3E50",
    "accent":    "#2980B9",
    "highlight": "#E67E22",
    "neutral":   "#7F8C8D",
}


def _bold_ticks(ax):
    """Helper to forcefully bold X and Y axis tick labels."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _save(fig, name):
    # Bold ticks for every axes in the figure before saving
    for ax in fig.get_axes():
        _bold_ticks(ax)
        
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")


def _title(ax, text):
    ax.set_title(text, fontsize=16, fontweight="bold", pad=12)


def _suptitle(fig, text, y=1.02):
    fig.suptitle(text, fontsize=18, fontweight="bold", y=y)


def _require(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"\n❌ Missing {label}:\n   {path}\n   Project root: {ROOT}"
        )


# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
print("\nLoading data...")

_require(PROCESSED / "X_test.csv",         "processed test features")
_require(PROCESSED / "X_test_scaled.csv",  "scaled test features")
_require(PROCESSED / "y_test.csv",         "test labels")
_require(MODEL_DIR / "split_metadata.json","split metadata")
_require(MODEL_DIR / "model_metadata.json","model metadata")
_require(MODEL_DIR / "model_metrics.json", "model metrics")

X_test        = pd.read_csv(PROCESSED / "X_test.csv")
X_test_scaled = pd.read_csv(PROCESSED / "X_test_scaled.csv")
y_test        = pd.read_csv(PROCESSED / "y_test.csv").squeeze()

with open(MODEL_DIR / "split_metadata.json") as f:
    split_meta = json.load(f)
with open(MODEL_DIR / "model_metadata.json") as f:
    model_meta = json.load(f)
with open(MODEL_DIR / "model_metrics.json") as f:
    model_metrics = json.load(f)

smote_csv = PROCESSED / "y_train_resampled.csv"
y_train_res = pd.read_csv(smote_csv).squeeze() if smote_csv.exists() else None

df_raw = None
if RAW_DATA.exists():
    df_raw = pd.read_csv(RAW_DATA)
    print(f"  Raw dataset: {df_raw.shape}")

print(f"  Test set: {X_test.shape} | Fraud: {y_test.sum():,}")


# ══════════════════════════════════════════════════════════════
# PART A — EDA FIGURES (5)
# ══════════════════════════════════════════════════════════════

def figure_eda_distribution():
    if df_raw is None: return
    print("\n[EDA] Target distribution...")
    counts = df_raw["isFraud"].value_counts().sort_index()
    total = counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bars = axes[0].bar(["Non-Fraud", "Fraud"], counts.values,
                       color=[PALETTE["nonfraud"], PALETTE["fraud"]],
                       edgecolor="black", linewidth=1.8)
    axes[0].set_ylabel("Transaction Count")
    _title(axes[0], "Fraud vs Non-Fraud Count")
    for b, v in zip(bars, counts.values):
        axes[0].text(b.get_x() + b.get_width() / 2, v,
                     f"{v:,}\n({100 * v / total:.3f}%)",
                     ha="center", va="bottom",
                     fontsize=13, fontweight="bold")

    axes[1].pie(counts.values, labels=["Non-Fraud", "Fraud"],
                autopct="%1.3f%%",
                colors=[PALETTE["nonfraud"], PALETTE["fraud"]],
                explode=(0.03, 0.03), startangle=90,
                textprops={"fontsize": 13, "fontweight": "bold"},
                wedgeprops={"edgecolor": "black", "linewidth": 1.8})
    _title(axes[1], "Fraud Proportion")
    fig.tight_layout()
    _save(fig, "fig01_fraud_distribution.png")


def figure_eda_type_fraud():
    if df_raw is None: return
    print("\n[EDA] Fraud by transaction type...")
    df_raw["type"] = df_raw["type"].astype(str)
    stats = df_raw.groupby("type")["isFraud"].agg(["sum", "count", "mean"])
    stats.columns = ["Fraud_Count", "Total", "Rate"]
    stats["Rate"] *= 100
    stats = stats.sort_values("Fraud_Count", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(stats.index, stats["Fraud_Count"],
                  color=PALETTE["fraud"], edgecolor="black", linewidth=1.8)
    ax.set_ylabel("Fraud Transactions")
    _title(ax, "Fraud Cases by Transaction Type")
    for b, v in zip(bars, stats["Fraud_Count"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig02_transaction_type_fraud.png")


def figure_eda_amount():
    if df_raw is None: return
    print("\n[EDA] Amount analysis...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].hist(df_raw["amount"], bins=50, color=PALETTE["accent"],
                 edgecolor="black", alpha=0.85, linewidth=1.2)
    axes[0].set_xlabel("Amount"); axes[0].set_ylabel("Frequency")
    _title(axes[0], "Raw Amount Distribution")

    axes[1].hist(np.log1p(df_raw["amount"]), bins=50,
                 color=PALETTE["highlight"], edgecolor="black",
                 alpha=0.85, linewidth=1.2)
    axes[1].set_xlabel("log(Amount + 1)"); axes[1].set_ylabel("Frequency")
    _title(axes[1], "Log-Transformed Distribution")

    fraud_amt = df_raw.loc[df_raw["isFraud"] == 1, "amount"]
    nonfraud_amt = df_raw.loc[df_raw["isFraud"] == 0, "amount"]
    bp = axes[2].boxplot([nonfraud_amt, fraud_amt],
                         tick_labels=["Non-Fraud", "Fraud"],
                         patch_artist=True, widths=0.6,
                         medianprops={"color": "black", "linewidth": 2},
                         whiskerprops={"linewidth": 1.5},
                         capprops={"linewidth": 1.5},
                         boxprops={"linewidth": 1.5})
    for patch, color in zip(bp["boxes"],
                            [PALETTE["nonfraud"], PALETTE["fraud"]]):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    axes[2].set_yscale("log"); axes[2].set_ylabel("Amount (log)")
    _title(axes[2], "Amount by Fraud Status")
    fig.tight_layout()
    _save(fig, "fig03_amount_analysis.png")


def figure_eda_time():
    if df_raw is None: return
    print("\n[EDA] Time analysis...")
    df_raw["hour"] = df_raw["step"] % 24

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    df_raw.groupby("hour").size().plot(ax=axes[0],
                                       color=PALETTE["accent"], lw=2.4)
    axes[0].set_xlabel("Hour of Day"); axes[0].set_ylabel("Transaction Count")
    _title(axes[0], "All Transactions by Hour")

    df_raw.loc[df_raw["isFraud"] == 1].groupby("hour").size().plot(
        ax=axes[1], color=PALETTE["fraud"], lw=2.4)
    axes[1].set_xlabel("Hour of Day"); axes[1].set_ylabel("Fraud Count")
    _title(axes[1], "Fraud Transactions by Hour")
    fig.tight_layout()
    _save(fig, "fig04_time_analysis.png")


def figure_eda_correlation():
    if df_raw is None: return
    print("\n[EDA] Correlation heatmap...")
    num_df = df_raw.select_dtypes(include=[np.number])
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax,
                annot_kws={"size": 11, "weight": "bold"},
                cbar_kws={"label": "Pearson r", "shrink": 0.8},
                linewidths=0.5, linecolor="white")
    ax.figure.axes[-1].yaxis.label.set_size(13)
    ax.figure.axes[-1].yaxis.label.set_weight("bold")
    _title(ax, "Feature Correlation Matrix")
    fig.tight_layout()
    _save(fig, "fig05_correlation_matrix.png")


# ══════════════════════════════════════════════════════════════
# PART B — SMOTE FIGURE
# ══════════════════════════════════════════════════════════════

def figure_smote_effect():
    print("\n[FE] SMOTE effect figure...")

    n_fraud_before    = int(split_meta["train_fraud"])
    n_train_before    = int(split_meta["train_size"])
    n_nonfraud_before = n_train_before - n_fraud_before

    n_fraud_after     = int(split_meta["train_fraud_after_smote"])
    n_train_after     = int(split_meta["train_resampled_size"])
    n_nonfraud_after  = n_train_after - n_fraud_after

    if y_train_res is not None:
        actual_after = int(y_train_res.sum())
        if actual_after != n_fraud_after:
            n_fraud_after    = actual_after
            n_nonfraud_after = int((y_train_res == 0).sum())

    print(f"  Before : Non-Fraud={n_nonfraud_before:,} | Fraud={n_fraud_before:,}")
    print(f"  After  : Non-Fraud={n_nonfraud_after:,} | Fraud={n_fraud_after:,}")

    ratio_before = n_nonfraud_before / max(n_fraud_before, 1)
    ratio_after  = n_nonfraud_after  / max(n_fraud_after,  1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = ["Non-Fraud", "Fraud"]
    colors = [PALETTE["nonfraud"], PALETTE["fraud"]]

    for ax, vals, ratio, label in [
        (axes[0], [n_nonfraud_before, n_fraud_before], ratio_before, "Before SMOTE"),
        (axes[1], [n_nonfraud_after,  n_fraud_after],  ratio_after,  "After SMOTE"),
    ]:
        bars = ax.bar(labels, vals, color=colors,
                      edgecolor="black", linewidth=1.8)
        ax.set_yscale("log")
        ax.set_ylabel("Sample Count (log scale)")
        _title(ax, f"{label}  (Ratio 1:{ratio:.0f})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:,}", ha="center", va="bottom",
                    fontsize=13, fontweight="bold")

    _suptitle(fig, "Effect of SMOTE Oversampling on Training Set")
    fig.tight_layout()
    _save(fig, "fig06_smote_effect.png")


# ══════════════════════════════════════════════════════════════
# PART C — MODEL TRAINING FIGURES
# ══════════════════════════════════════════════════════════════

def figure_model_comparison():
    print("\n[Model] Model comparison...")
    results_df = pd.DataFrame(model_metrics).T
    results_df = results_df[["precision", "recall", "f1",
                             "roc_auc", "pr_auc", "mcc"]].astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    s = results_df["f1"].sort_values()
    s.plot(kind="barh", ax=axes[0, 0], color=PALETTE["accent"],
           edgecolor="black", linewidth=1.4)
    _title(axes[0, 0], "F1 Score by Model")
    axes[0, 0].set_xlim(0, 1)
    for i, v in enumerate(s.values):
        axes[0, 0].text(v + 0.01, i, f"{v:.3f}", va="center",
                        fontsize=12, fontweight="bold")

    results_df[["precision", "recall"]].plot(
        kind="bar", ax=axes[0, 1],
        color=[PALETTE["accent"], PALETTE["fraud"]],
        edgecolor="black", linewidth=1.2,
    )
    _title(axes[0, 1], "Precision vs Recall")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis="x", rotation=45)

    s = results_df["pr_auc"].sort_values()
    s.plot(kind="barh", ax=axes[1, 0], color=PALETTE["nonfraud"],
           edgecolor="black", linewidth=1.4)
    _title(axes[1, 0], "PR-AUC Comparison")
    axes[1, 0].set_xlim(0, 1)
    for i, v in enumerate(s.values):
        axes[1, 0].text(v + 0.01, i, f"{v:.3f}", va="center",
                        fontsize=12, fontweight="bold")

    s = results_df["mcc"].sort_values()
    s.plot(kind="barh", ax=axes[1, 1], color=PALETTE["highlight"],
           edgecolor="black", linewidth=1.4)
    _title(axes[1, 1], "Matthews Correlation Coefficient (MCC)")
    axes[1, 1].set_xlim(-1, 1)
    for i, v in enumerate(s.values):
        axes[1, 1].text(v + 0.01, i, f"{v:.3f}", va="center",
                        fontsize=12, fontweight="bold")

    _suptitle(fig, "Model Performance Comparison (Validation-Tuned Threshold)",
              y=1.00)
    fig.tight_layout()
    _save(fig, "fig07_model_comparison.png")


def figure_roc_pr():
    print("\n[Model] ROC / PR curves...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    linear_models = {"Logistic Regression", "SVM"}
    cmap = plt.get_cmap("tab10")

    for i, (name, metrics) in enumerate(model_metrics.items()):
        safe = name.lower().replace(" ", "_")
        path = MODEL_DIR / f"{safe}.pkl"
        if not path.exists():
            continue
        model = joblib.load(path)
        X_eval = X_test_scaled if name in linear_models else X_test
        try:
            y_proba = model.predict_proba(X_eval)[:, 1]
        except Exception:
            continue

        color = cmap(i % 10)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[0].plot(fpr, tpr, lw=2.4, color=color,
                     label=f"{name} (AUC={metrics['roc_auc']:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        axes[1].plot(rec, prec, lw=2.4, color=color,
                     label=f"{name} (PR-AUC={metrics['pr_auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC=0.5)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    _title(axes[0], "ROC Curves")
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    _title(axes[1], "Precision-Recall Curves")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    _save(fig, "fig08_roc_pr_curves.png")


# ══════════════════════════════════════════════════════════════
# PART D — BUSINESS EVALUATION
# ══════════════════════════════════════════════════════════════

def figure_cost_threshold():
    print("\n[Eval] Cost-threshold curve...")
    best_name = model_meta["best_model"]
    safe = best_name.lower().replace(" ", "_")
    model = joblib.load(MODEL_DIR / f"{safe}.pkl")
    X_eval = X_test_scaled if best_name in {"Logistic Regression", "SVM"} else X_test

    y_proba = model.predict_proba(X_eval)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 100)
    costs, f1s = [], []
    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
        costs.append(fn * 250 + fp)
        f1s.append(2 * tp / (2 * tp + fp + fn + 1e-10))

    best_idx = int(np.argmin(costs))

    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax1.plot(thresholds, costs, color=PALETTE["accent"], lw=2.8,
             label="Total Cost (BDT)")
    ax1.axvline(thresholds[best_idx], color=PALETTE["fraud"],
                ls="--", lw=2.0,
                label=f"Cost-optimal = {thresholds[best_idx]:.3f}")
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Total Business Cost (BDT)", color=PALETTE["accent"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["accent"])

    ax2 = ax1.twinx()
    ax2.plot(thresholds, f1s, color=PALETTE["nonfraud"], alpha=0.65,
             lw=2.4, label="F1 Score")
    ax2.set_ylabel("F1 Score", color=PALETTE["nonfraud"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["nonfraud"])
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper center", ncol=3, prop={"size": 12, "weight": "bold"})

    _title(ax1, f"Threshold vs Cost & F1 ({best_name}) — MFS 250:1 Cost Ratio")
    fig.tight_layout()
    _save(fig, "fig09_cost_threshold_curve.png")


def figure_confusion_matrix():
    print("\n[Eval] Confusion matrix...")
    best_name = model_meta["best_model"]
    safe = best_name.lower().replace(" ", "_")
    model = joblib.load(MODEL_DIR / f"{safe}.pkl")
    X_eval = X_test_scaled if best_name in {"Logistic Regression", "SVM"} else X_test

    y_proba = model.predict_proba(X_eval)[:, 1]
    threshold = float(model_metrics[best_name]["threshold"])
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                cbar=False,
                xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"],
                annot_kws={"size": 17, "weight": "bold"},
                linewidths=2.5, linecolor="white")
    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold", labelpad=10)
    _title(ax, f"Confusion Matrix — {best_name} (τ = {threshold:.3f})")
    fig.tight_layout()
    _save(fig, "fig10_confusion_matrix.png")


# ══════════════════════════════════════════════════════════════
# PART E — SHAP INTERPRETABILITY
# ══════════════════════════════════════════════════════════════

def build_shap_explainer():
    print("\n[SHAP] Building explainer...")
    best_name = model_meta["best_model"]
    safe = best_name.lower().replace(" ", "_")
    best_model = joblib.load(MODEL_DIR / f"{safe}.pkl")

    X_te = X_test_scaled if best_name in {"Logistic Regression", "SVM"} else X_test

    fraud_idx = y_test[y_test == 1].sample(500, random_state=RANDOM_STATE).index
    nonfraud_idx = y_test[y_test == 0].sample(500, random_state=RANDOM_STATE).index
    sample_idx = fraud_idx.union(nonfraud_idx)

    X_sample = X_te.loc[sample_idx].reset_index(drop=True)
    y_sample = y_test.loc[sample_idx].reset_index(drop=True)
    print(f"  SHAP sample: {X_sample.shape}")

    model_obj = best_model
    if hasattr(best_model, "calibrated_classifiers_"):
        model_obj = best_model.calibrated_classifiers_[0].estimator

    try:
        explainer = shap.TreeExplainer(model_obj)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        bg = shap.sample(X_sample, 50, random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(
            lambda x: best_model.predict_proba(x)[:, 1], bg)
        shap_values = explainer.shap_values(X_sample.iloc[:300], nsamples=100)
        X_sample = X_sample.iloc[:300].reset_index(drop=True)
        y_sample = y_sample.iloc[:300].reset_index(drop=True)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)) and not np.isscalar(exp_val):
        exp_val = float(exp_val[1])
    else:
        exp_val = float(exp_val)

    print(f"  Base value: {exp_val:.4f}")
    return best_model, X_sample, y_sample, sv, exp_val, best_name


def figure_shap_bar(X_sample, sv, best_name):
    print("\n[SHAP] Global bar plot...")
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 13,
                         "axes.titlesize": 16,
                         "axes.titleweight": "bold",
                         "axes.labelsize": 13}):
        plt.figure(figsize=(12, 9))
        shap.summary_plot(sv, X_sample, plot_type="bar",
                          show=False, max_display=20)
        plt.title(f"SHAP Feature Importance — {best_name}",
                  fontsize=16, fontweight="bold", pad=12)
        plt.xlabel("Mean |SHAP value|", fontsize=13, fontweight="bold")
        plt.tight_layout()
        _save(plt.gcf(), "fig11_shap_bar.png")


def figure_shap_beeswarm(X_sample, sv, best_name):
    print("\n[SHAP] Beeswarm plot...")
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 13,
                         "axes.titlesize": 16,
                         "axes.titleweight": "bold",
                         "axes.labelsize": 13}):
        plt.figure(figsize=(12, 11))
        shap.summary_plot(sv, X_sample, show=False, max_display=20)
        plt.title(f"SHAP Summary — {best_name}",
                  fontsize=16, fontweight="bold", pad=12)
        plt.xlabel("SHAP value (impact on model output)",
                   fontsize=13, fontweight="bold")
        plt.tight_layout()
        _save(plt.gcf(), "fig12_shap_beeswarm.png")


def figure_shap_dependence(X_sample, sv):
    print("\n[SHAP] Dependence plots (top 4)...")
    mean_abs = np.abs(sv).mean(axis=0)
    top_features = list(X_sample.columns[np.argsort(mean_abs)[-4:][::-1]])
    print(f"  Top 4: {top_features}")

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 12,
                         "axes.titlesize": 14,
                         "axes.titleweight": "bold",
                         "axes.labelsize": 12}):
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        axes = axes.ravel()

        for i, feature in enumerate(top_features):
            try:
                shap.dependence_plot(feature, sv, X_sample,
                                     ax=axes[i], show=False)
                axes[i].set_title(f"SHAP Dependence: {feature}",
                                  fontsize=14, fontweight="bold")
            except Exception:
                axes[i].text(0.5, 0.5, f"Failed: {feature}",
                             ha="center", va="center",
                             transform=axes[i].transAxes,
                             fontsize=13, fontweight="bold")

        fig.tight_layout()
        _save(fig, "fig13_shap_dependence.png")


def figure_shap_waterfalls(best_model, X_sample, y_sample, sv, exp_val):
    print("\n[SHAP] Waterfall plots...")

    y_proba = best_model.predict_proba(X_sample)[:, 1]
    best_name = model_meta["best_model"]
    threshold = float(model_metrics[best_name]["threshold"])
    y_pred = (y_proba >= threshold).astype(int)

    fraud_pos = np.where(y_sample.values == 1)[0]
    tp_pos = [p for p in fraud_pos if y_pred[p] == 1]
    fn_pos = [p for p in fraud_pos if y_pred[p] == 0]

    def _waterfall(idx, title, fname):
        try:
            exp = shap.Explanation(
                values=sv[idx],
                base_values=exp_val,
                data=X_sample.iloc[idx].values,
                feature_names=X_sample.columns.tolist(),
            )
            with plt.rc_context({"font.family": "sans-serif",
                                 "font.sans-serif": ["Arial", "DejaVu Sans"],
                                 "font.size": 12,
                                 "axes.titlesize": 14,
                                 "axes.titleweight": "bold",
                                 "axes.labelsize": 12}):
                plt.figure(figsize=(12, 7))
                shap.plots.waterfall(exp, max_display=15, show=False)
                plt.title(title, fontsize=14, fontweight="bold", pad=10)
                plt.tight_layout()
                _save(plt.gcf(), fname)
        except Exception as e:
            print(f"  ✗ Waterfall failed: {e}")

    if tp_pos:
        idx = tp_pos[0]
        _waterfall(idx,
                   f"SHAP Waterfall — Caught Fraud (P={y_proba[idx]:.3f})",
                   "fig14_shap_waterfall_tp.png")
    if fn_pos:
        idx = fn_pos[0]
        _waterfall(idx,
                   f"SHAP Waterfall — Missed Fraud (P={y_proba[idx]:.3f})",
                   "fig15_shap_waterfall_fn.png")


def figure_shap_top_features_bar(X_sample, sv):
    print("\n[SHAP] Top-feature bar chart...")
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(14, 10))

    bars = ax.barh(
        imp["feature"], imp["mean_abs_shap"],
        color=PALETTE["accent"],
        edgecolor="black",
        linewidth=1.6,
        height=0.7,
    )

    ax.set_xlabel("Mean |SHAP value|", fontsize=18, fontweight="bold", labelpad=12)
    ax.set_ylabel("")

    ax.set_title(
        "Top-15 Features by Mean Absolute SHAP Value",
        fontsize=20, fontweight="bold", pad=16,
    )

    # Very large, bold ticks for this specific chart
    ax.tick_params(axis="y", labelsize=15)
    ax.tick_params(axis="x", labelsize=14)

    x_max = imp["mean_abs_shap"].max()
    for b, v in zip(bars, imp["mean_abs_shap"]):
        ax.text(
            v + x_max * 0.012,
            b.get_y() + b.get_height() / 2,
            f"{v:.4f}",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_xlim(0, x_max * 1.15)

    ax.grid(True, axis="x", alpha=0.35, linestyle="--", linewidth=0.8)
    ax.grid(False, axis="y")

    fig.tight_layout()
    _save(fig, "fig16_shap_top_features.png")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Generating ALL publication-grade figures")
    print("=" * 60)

    figure_eda_distribution()
    figure_eda_type_fraud()
    figure_eda_amount()
    figure_eda_time()
    figure_eda_correlation()

    figure_smote_effect()

    figure_model_comparison()
    figure_roc_pr()

    figure_cost_threshold()
    figure_confusion_matrix()

    print("\n" + "─" * 60)
    print("Loading SHAP — this may take ~1 minute...")
    print("─" * 60)

    best_model, X_sample, y_sample, sv, exp_val, best_name = build_shap_explainer()

    figure_shap_bar(X_sample, sv, best_name)
    figure_shap_beeswarm(X_sample, sv, best_name)
    figure_shap_dependence(X_sample, sv)
    figure_shap_waterfalls(best_model, X_sample, y_sample, sv, exp_val)
    figure_shap_top_features_bar(X_sample, sv)

    print("\n" + "=" * 60)
    print(f"✅ ALL 16 FIGURES SAVED → {FIG_DIR}")
    print("=" * 60)
    for f in sorted(FIG_DIR.glob("fig*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name:42s} {size_kb:8.1f} KB")