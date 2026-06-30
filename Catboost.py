import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import warnings

# Fix matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (RandomizedSearchCV, train_test_split,
                                     StratifiedKFold, KFold)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import randint, uniform
from catboost import cv as catboost_cv
from catboost import CatBoostRegressor, Pool
import shap

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH       = 'DF1.csv'

TARGET_VARS    = ['CO2_conv', 'C5plus_sel', 'C5plus_yield']
PRIMARY_TARGET = 'C5plus_yield'   # used for stratified binning

# All catalytic output columns — excluded from features to prevent leakage
CATALYTIC_OUTPUT_FEATURES = [
    'CO2_conv', 'CO_yield', 'HC_yield',
    'C1_sel', 'C2C4_sel', 'C5plus_sel',
    'C1_yield', 'C2C4_yield', 'C5plus_yield',
]

CATEGORICAL_COLS = ['support', 'synth_steps', 'synth']

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

print("="*80)
print("CATBOOST MODEL TRAINING WITH SHAP ANALYSIS")
print("="*80)

# Create output directories
os.makedirs('outputs/catboost/models', exist_ok=True)
os.makedirs('outputs/catboost/figures', exist_ok=True)
os.makedirs('outputs/catboost/figures/shap', exist_ok=True)
os.makedirs('outputs/catboost/predictions', exist_ok=True)
os.makedirs('outputs/catboost/results', exist_ok=True)


# ============================================================================
# DATA LOADING, SPLITTING AND CV GENERATION
# ============================================================================

print(f"\n📂 Loading dataset from {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
print(f"  ✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Leakage guard: drop all catalytic outputs from features ────────────────
available_targets = [t for t in TARGET_VARS if t in df.columns]
if len(available_targets) < len(TARGET_VARS):
    print(f"  ⚠️  Missing target columns: {set(TARGET_VARS) - set(available_targets)}")

cols_to_drop = [c for c in CATALYTIC_OUTPUT_FEATURES if c in df.columns]
X_full      = df.drop(columns=cols_to_drop)
y_dict_full = {t: df[t] for t in available_targets}

existing_cats = [c for c in CATEGORICAL_COLS if c in X_full.columns]
num_features  = [c for c in X_full.columns if c not in existing_cats]

print(f"\n  Leakage prevention : dropped {len(cols_to_drop)} catalytic output columns")
print(f"  Remaining features : {X_full.shape[1]} ({len(num_features)} numerical + {len(existing_cats)} categorical)")
print(f"  Targets            : {available_targets}")
print(f"  Samples            : {len(X_full)}")

# ── Stratified train / test split ─────────────────────────────────────────
print(f"\n  Performing stratified {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split ...")

try:
    strat_bins = pd.qcut(y_dict_full[PRIMARY_TARGET], q=5, labels=False, duplicates='drop')
    print(f"  ✓ Stratification: 5 quantile bins on '{PRIMARY_TARGET}'")
except Exception:
    strat_bins = None
    print(f"  ⚠️  Stratification binning failed — using random split")

split_kwargs = dict(test_size=TEST_SIZE, random_state=RANDOM_STATE)
if strat_bins is not None:
    split_kwargs['stratify'] = strat_bins

X_train, X_test, idx_train, idx_test = train_test_split(
    X_full, X_full.index, **split_kwargs
)
y_train_dict = {t: y_dict_full[t].loc[idx_train] for t in available_targets}
y_test_dict  = {t: y_dict_full[t].loc[idx_test]  for t in available_targets}

print(f"  ✓ Train: {len(X_train)} samples | Test: {len(X_test)} samples")

print(f"\n  Target distributions (train vs test):")
for t in available_targets:
    tr, te = y_train_dict[t].mean(), y_test_dict[t].mean()
    print(f"    {t:20s}: Train={tr:.2f}, Test={te:.2f}, Diff={abs(tr-te):.2f}")

# ── 5-fold CV indices ──────────────────────────────────────────────────────
print(f"\n  Generating {CV_FOLDS}-fold CV indices ...")
if strat_bins is not None:
    cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_splits   = list(cv_splitter.split(X_train, strat_bins.loc[idx_train]))
else:
    cv_splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_splits   = list(cv_splitter.split(X_train))

for i, (tr_idx, val_idx) in enumerate(cv_splits):
    print(f"    Fold {i+1}: Train={len(tr_idx)}, Val={len(val_idx)}")

# ── Sanity checks ──────────────────────────────────────────────────────────
feature_names  = list(X_train.columns)
leaked_targets = set(available_targets) & set(feature_names)
leaked_outputs = set(CATALYTIC_OUTPUT_FEATURES) & set(feature_names)
overlap        = set(idx_train) & set(idx_test)
all_cv_indices = set(i for tr, val in cv_splits for i in [*tr, *val])
missing_folds  = set(range(len(X_train))) - all_cv_indices

print(f"\n  ── Pre-flight checks ──────────────────────────────────────")
print(f"  Target leakage   : {'❌ ERROR — ' + str(leaked_targets) if leaked_targets else '✓ OK'}")
print(f"  Output leakage   : {'❌ ERROR — ' + str(leaked_outputs) if leaked_outputs else '✓ OK'}")
print(f"  Train/test overlap: {'❌ ERROR — ' + str(len(overlap)) + ' samples' if overlap else '✓ OK'}")
print(f"  CV fold coverage : {'⚠️  WARNING — ' + str(len(missing_folds)) + ' missing' if missing_folds else '✓ OK'}")

if leaked_targets or leaked_outputs:
    raise ValueError("Data leakage detected — aborting training.")

# ── CatBoost native categorical handling (no LabelEncoder needed) ──────────
# CatBoost accepts raw string categoricals — just ensure str type and no NAs.
# [CAT-FIX] Detect categoricals by DTYPE (object/category) and union with the
# known CATEGORICAL_COLS list. The old code used the hardcoded list only, so a
# string column not in the list (e.g. synth1/synth2/synth3) would be passed to
# CatBoost as numeric and raise a "cannot convert to float" error or be mishandled.
# All other scripts (XGB/RF/GBDT/GPR) already detect by dtype — this aligns CatBoost.
dtype_cats = [c for c in X_train.columns
              if X_train[c].dtype == object or str(X_train[c].dtype) == 'category']
named_cats = [c for c in CATEGORICAL_COLS if c in X_train.columns]
cat_cols   = sorted(set(dtype_cats) | set(named_cats),
                    key=lambda c: X_train.columns.tolist().index(c))

for col in cat_cols:
    X_train[col] = X_train[col].fillna('missing').astype(str)
    X_test[col]  = X_test[col].fillna('missing').astype(str)

cat_feature_indices = [X_train.columns.tolist().index(col) for col in cat_cols]

print(f"\n✓ CatBoost native categoricals : {cat_cols}")
print(f"  Column indices               : {cat_feature_indices}")
print(f"✓ Ready — {len(X_train)} train | {len(X_test)} test | {len(feature_names)} features | {CV_FOLDS}-fold CV")

# Align TARGET_VARS to what is actually present in the CSV
TARGET_VARS = available_targets


# HELPER FUNCTIONS
def calculate_mape(y_true, y_pred, threshold=0.01):
    """Calculate MAPE with proper handling of zero/near-zero values."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask   = np.abs(y_true) > threshold
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(model, X_train, X_test, y_train, y_test, return_predictions=False):
    """Comprehensive model evaluation with all metrics."""
    y_pred_train = np.array(model.predict(X_train))
    y_pred_test  = np.array(model.predict(X_test))
    y_train      = np.array(y_train)
    y_test       = np.array(y_test)

    metrics = {
        'Train_R2':   r2_score(y_train, y_pred_train),
        'Test_R2':    r2_score(y_test,  y_pred_test),
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test_RMSE':  np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        'Train_MAE':  mean_absolute_error(y_train, y_pred_train),
        'Test_MAE':   mean_absolute_error(y_test,  y_pred_test),
    }
    metrics['Train_MAPE'] = calculate_mape(y_train, y_pred_train)
    metrics['Test_MAPE']  = calculate_mape(y_test,  y_pred_test)
    metrics['Gap']        = metrics['Train_R2'] - metrics['Test_R2']

    if return_predictions:
        return metrics, y_pred_train, y_pred_test
    return metrics


# [CV-FIX] Proper K-fold CV returning the MEAN and STD of per-fold R².
# The old version derived a single pooled R² from CatBoost's aggregated
# test-RMSE and reported `RMSE-std / sqrt(var)` as the "std" — that is not a
# valid R² standard deviation and is not comparable to the
# cross_val_score(scoring='r2') used by the RF / GBDT / GPR scripts. Here each
# fold trains on its own split (no test leakage) and is scored with r2_score on
# its held-out fold, exactly like the other models.
def catboost_cross_val_r2(params, X, y, fold_count):
    try:
        strat    = pd.qcut(y, q=5, labels=False, duplicates='drop')
        splitter = StratifiedKFold(n_splits=fold_count, shuffle=True,
                                   random_state=RANDOM_STATE).split(X, strat)
    except Exception:
        splitter = KFold(n_splits=fold_count, shuffle=True,
                         random_state=RANDOM_STATE).split(X)

    fold_params = {k: v for k, v in params.items()}
    fold_params.setdefault('loss_function', 'RMSE')
    fold_params['eval_metric']   = 'RMSE'
    fold_params['random_seed']   = RANDOM_STATE
    fold_params['logging_level'] = 'Silent'
    fold_params['thread_count']  = -1
    # fixed iterations per fold (no inner early stopping → no eval_set needed)
    fold_params.pop('use_best_model', None)
    fold_params.pop('early_stopping_rounds', None)

    fold_r2 = []
    for tr_idx, val_idx in splitter:
        X_ftr, X_fval = X.iloc[tr_idx], X.iloc[val_idx]
        y_ftr, y_fval = y.iloc[tr_idx], y.iloc[val_idx]
        m = CatBoostRegressor(**fold_params)
        m.fit(Pool(X_ftr, label=y_ftr, cat_features=cat_feature_indices),
              verbose=False)
        fold_r2.append(r2_score(np.asarray(y_fval),
                                np.asarray(m.predict(X_fval))))
    return float(np.mean(fold_r2)), float(np.std(fold_r2))


def plot_predictions(y_test, y_train, y_train_pred, y_test_pred, target, save_path):
    """Enhanced parity plot with comprehensive statistics."""

    def _metrics(y_true, y_hat):
        return (r2_score(y_true, y_hat),
                np.sqrt(mean_squared_error(y_true, y_hat)),
                mean_absolute_error(y_true, y_hat),
                calculate_mape(y_true, y_hat))

    tr_r2, tr_rmse, tr_mae, tr_mape = _metrics(np.array(y_train), np.array(y_train_pred))
    te_r2, te_rmse, te_mae, te_mape = _metrics(np.array(y_test),  np.array(y_test_pred))

    all_true = np.concatenate([np.asarray(y_train).ravel(), np.asarray(y_test).ravel()])
    all_pred = np.concatenate([np.asarray(y_train_pred).ravel(), np.asarray(y_test_pred).ravel()])
    min_val  = float(min(all_true.min(), all_pred.min()))
    max_val  = float(max(all_true.max(), all_pred.max()))
    pad      = 0.05 * (max_val - min_val + 1e-12)
    lo, hi   = min_val - pad, max_val + pad

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_train, y_train_pred, alpha=0.4, s=50, marker="o",
               edgecolors="k", linewidth=0.5, label=f"Train (n={len(y_train)})", c='skyblue')
    ax.scatter(y_test,  y_test_pred,  alpha=0.7, s=70, marker="^",
               edgecolors="k", linewidth=0.6, label=f"Test (n={len(y_test)})",   c='coral')
    ax.plot([lo, hi], [lo, hi], "r--", lw=2.5, label="Perfect prediction", alpha=0.8)
    ax.fill_between([lo, hi], [lo*0.9, hi*0.9], [lo*1.1, hi*1.1],
                    alpha=0.1, color='gray', label='±10% error band')

    mape_tr = f"{tr_mape:.2f}%" if not np.isnan(tr_mape) else "N/A"
    mape_te = f"{te_mape:.2f}%" if not np.isnan(te_mape) else "N/A"
    textstr = (f"TRAIN (n={len(y_train)})\n"
               f"  R² = {tr_r2:.4f}\n  RMSE = {tr_rmse:.4f}\n"
               f"  MAE = {tr_mae:.4f}\n  MAPE = {mape_tr}\n\n"
               f"TEST (n={len(y_test)})\n"
               f"  R² = {te_r2:.4f}\n  RMSE = {te_rmse:.4f}\n"
               f"  MAE = {te_mae:.4f}\n  MAPE = {mape_te}")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7), family='monospace')

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Actual {target}",    fontsize=13, fontweight='bold')
    ax.set_ylabel(f"Predicted {target}", fontsize=13, fontweight='bold')
    ax.set_title(f"CatBoost – {target} Predictions", fontsize=15, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_learning_curves(model, X_train, y_train, cv_folds, target, save_path):
    """
    Manual learning curve using CatBoost native CV.
    FIX: replaces sklearn learning_curve() which triggers __sklearn_tags__ error
    on CatBoost <1.2.7 with sklearn >=1.6.
    """
    params = model.get_params()
    # Strip params that aren't valid standalone CatBoost training params
    for key in ['early_stopping_rounds', 'eval_metric', 'use_best_model', 'cat_features']:
        params.pop(key, None)
    params.update({
        'eval_metric': 'RMSE',
        'random_seed': RANDOM_STATE,
        'verbose':     0,
        'iterations':  params.get('iterations', 200),
    })

    train_sizes = np.linspace(0.2, 1.0, 5)   # ← 8 points → 5, skip tiny subsets
    n_samples   = len(X_train)
    train_r2s, val_r2s = [], []

    # Single fixed val split — same val set for all size points for fair comparison
    val_size = max(20, int(n_samples * 0.2))
    rng_lc   = np.random.RandomState(RANDOM_STATE)
    all_idx  = np.arange(n_samples)
    val_idx  = rng_lc.choice(all_idx, val_size, replace=False)
    train_pool_idx = np.setdiff1d(all_idx, val_idx)

    X_val_lc = X_train.iloc[val_idx]
    y_val_lc = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]

    for frac in train_sizes:
        n        = max(10, int(len(train_pool_idx) * frac))
        sub_idx  = rng_lc.choice(train_pool_idx, n, replace=False)
        X_sub    = X_train.iloc[sub_idx]
        y_sub    = y_train.iloc[sub_idx] if hasattr(y_train, 'iloc') else y_train[sub_idx]

        lc_model = CatBoostRegressor(
            **{k: v for k, v in params.items()
               if k not in ['loss_function', 'eval_metric', 'random_seed', 'verbose']},
            random_seed=RANDOM_STATE,
            verbose=0,
            early_stopping_rounds=20,
            eval_metric='RMSE',
            use_best_model=True,
            thread_count=-1,
        )
        lc_model.fit(
            Pool(X_sub,    label=y_sub,    cat_features=cat_feature_indices),
            eval_set=Pool(X_val_lc, label=y_val_lc, cat_features=cat_feature_indices),
            verbose=False
        )
        train_r2s.append(r2_score(y_sub,    lc_model.predict(X_sub)))
        val_r2s.append(  r2_score(y_val_lc, lc_model.predict(X_val_lc)))

    sizes_abs = (train_sizes * n_samples).astype(int)
    train_r2s = np.array(train_r2s)
    val_r2s   = np.array(val_r2s)
    # No std bands — single split has no variance estimate

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(sizes_abs, train_r2s, label='Training Score',
            marker='o', linewidth=2.5, markersize=8, color='blue')
    ax.plot(sizes_abs, val_r2s,   label='Validation Score',
            marker='s', linewidth=2.5, markersize=8, color='orange')


    ax.set_xlabel('Training Set Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('R² Score',          fontsize=13, fontweight='bold')
    ax.set_title(f'Learning Curves – {target}', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, target, save_path, top_n=20):
    """Plot feature importance from trained CatBoost model."""
    importance_df = pd.DataFrame({
        'Feature':    list(feature_names),
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(importance_df)), importance_df['Importance'].values,
                   alpha=0.8, color='darkorange', edgecolor='black', linewidth=0.7)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (PredictionValuesChange)', fontsize=13, fontweight='bold')
    ax.set_title(f'Top {top_n} Features – {target}', fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    for bar in bars:
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height()/2,
                f'{w:.4f}', ha='left', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# SHAP HELPER FUNCTIONS
# ============================================================================

def compute_shap_values(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    shap_df   = pd.DataFrame(shap_vals, columns=feature_names)
    return explainer, shap_vals, shap_df


def plot_shap_summary(shap_vals, X, target, save_path, plot_type='dot', max_display=20):
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, plot_type=plot_type,
                      max_display=max_display, show=False, plot_size=None)
    title = (f"SHAP Summary (beeswarm) – {target}" if plot_type == 'dot'
             else f"SHAP Feature Importance (bar) – {target}")
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_shap_dependence(shap_vals, X, target, save_dir, top_n=5):
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_features  = np.argsort(mean_abs_shap)[::-1][:top_n]
    feat_names    = X.columns.tolist()
    for rank, feat_idx in enumerate(top_features, 1):
        feat_name = feat_names[feat_idx]
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.dependence_plot(feat_idx, shap_vals, X,
                             feature_names=feat_names, ax=ax, show=False)
        ax.set_title(f"SHAP Dependence – {feat_name}\n({target})",
                     fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        safe_feat = feat_name.replace('/', '_').replace(' ', '_')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"dep_rank{rank:02d}_{safe_feat}.png"),
                    dpi=200, bbox_inches='tight')
        plt.close()


def plot_shap_waterfall_mean(explainer, shap_vals, X, target, save_path):
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[0])
        
    # Initialize mean_data outside the if-block
    mean_data = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            mean_data.append(X[col].mean())
        else:
            mean_data.append(X[col].mode()[0])
            
    mean_data = np.array(mean_data, dtype=object)

    explanation = shap.Explanation(
        values=shap_vals.mean(axis=0),
        base_values=base_value,
        data=mean_data,
        feature_names=X.columns.tolist()
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=20, show=False)
    plt.title(f"SHAP Waterfall (mean explanation) – {target}",
              fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_shap_cross_target(shap_importance_dict, target_vars, save_path, top_n=15):
    top_feats_set = set()
    for target in target_vars:
        top_feats_set.update(shap_importance_dict[target].nlargest(top_n).index.tolist())
    top_feats = sorted(top_feats_set)

    df_plot = pd.DataFrame(
        {t: [shap_importance_dict[t].get(f, 0.0) for f in top_feats] for t in target_vars},
        index=top_feats
    )
    df_plot = df_plot.loc[df_plot.sum(axis=1).sort_values(ascending=False).index]

    n_targets = len(target_vars)
    fig, ax   = plt.subplots(figsize=(max(12, len(top_feats) * 0.6 + 2), 7))
    x         = np.arange(len(df_plot))
    width     = 0.8 / n_targets
    colors    = plt.cm.tab10(np.linspace(0, 0.9, n_targets))

    for i, (target, color) in enumerate(zip(target_vars, colors)):
        offset = (i - n_targets / 2 + 0.5) * width
        ax.bar(x + offset, df_plot[target].values, width,
               label=target, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Target SHAP Feature Importance Comparison',
                 fontsize=14, fontweight='bold', pad=14)
    ax.legend(title='Target', fontsize=10, title_fontsize=10,
              loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# MODEL TRAINING
# ============================================================================

print("\n" + "="*80)
print("STARTING MODEL TRAINING")
print("="*80)

baseline_results     = {}
tuned_results        = {}
best_params          = {}
tuning_decisions     = {}
final_models         = {}
final_results        = {}
search_times         = {}
shap_importance_dict = {}

for target in TARGET_VARS:
    print(f"\n{'='*80}")
    print(f"TARGET: {target}")
    print(f"{'='*80}")

    y_train = y_train_dict[target]
    y_test  = y_test_dict[target]

    # [LEAK-FIX] Early stopping must NOT see X_test. The old code fit both the
    # baseline and tuned models with eval_set=test_pool_t + use_best_model=True,
    # which selects the boosting iteration that minimises RMSE on the TEST set —
    # direct test-set leakage that inflates Test R² and makes CatBoost's numbers
    # non-comparable to the other models. We now carve a 15% internal validation
    # set from X_train only (mirrors the XGBoost V4 pipeline). X_test is used
    # exclusively for final scoring.
    X_tr_es, X_val_es, y_tr_es, y_val_es = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE
    )
    es_train_pool = Pool(X_tr_es,  label=y_tr_es,  cat_features=cat_feature_indices)
    es_val_pool   = Pool(X_val_es, label=y_val_es, cat_features=cat_feature_indices)

    # ========================================================================
    # STEP 1 — BASELINE MODEL
    # ========================================================================
    print("\n📊 Step 1: Training BASELINE model...")

    start_time  = datetime.now()
    cb_baseline = CatBoostRegressor(
        random_seed=RANDOM_STATE,
        verbose=0,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        early_stopping_rounds=50,
        eval_metric='RMSE',
        use_best_model=True,
        thread_count=-1,
    )
    cb_baseline.fit(es_train_pool, eval_set=es_val_pool, verbose=False)
    baseline_time = (datetime.now() - start_time).total_seconds()

    baseline_metrics = evaluate_model(cb_baseline, X_train, X_test, y_train, y_test)

    # FIX: use CatBoost native CV instead of sklearn cross_val_score
    r2_mean, r2_std = catboost_cross_val_r2(
        {
            'iterations':    cb_baseline.best_iteration_ if cb_baseline.best_iteration_ else 100,
            'learning_rate': 0.05,
            'depth':         6,
            'l2_leaf_reg':   3.0,
        },
        X_train, y_train, CV_FOLDS
    )
    baseline_metrics['CV_R2_Mean'] = r2_mean
    baseline_metrics['CV_R2_Std']  = r2_std
    baseline_results[target] = baseline_metrics

    print(f"  ✓ Baseline (trained in {baseline_time:.1f}s, "
          f"best_iter={cb_baseline.best_iteration_}):")
    print(f"    Train R²:  {baseline_metrics['Train_R2']:.4f}")
    print(f"    Test R²:   {baseline_metrics['Test_R2']:.4f}")
    print(f"    Gap:       {baseline_metrics['Gap']:.4f}")

    # ========================================================================
    # STEP 2 — HYPERPARAMETER TUNING
    # ========================================================================
    print("\n🔧 Step 2: Hyperparameter tuning with early stopping...")

    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    param_distributions = {
        'iterations':          randint(50, 500),
        'learning_rate':       uniform(0.01, 0.09),
        'depth':               randint(3, 8),
        'l2_leaf_reg':         uniform(1.0, 9.0),
        'random_strength':     uniform(0.5, 4.5),
        'bagging_temperature': uniform(0.0, 1.0),
        'border_count':        randint(32, 255),
        'min_data_in_leaf':    randint(1, 20),
    }

    INT_PARAMS = {'iterations', 'depth', 'border_count', 'min_data_in_leaf'}

    N_ITER      = 80
    rng         = np.random.RandomState(RANDOM_STATE)
    best_score  = -np.inf
    best_params_found = None
    n_failed          = 0

    start_time  = datetime.now()

    for i in range(N_ITER):
        sampled = {}
        for k, dist in param_distributions.items():
            val = dist.rvs(random_state=rng) 
            sampled[k] = int(val) if k in INT_PARAMS else float(val)

        try:
            cb_trial = CatBoostRegressor(
                **sampled,
                random_seed=RANDOM_STATE,
                logging_level='Silent',
                loss_function='RMSE',
                eval_metric='RMSE',
                early_stopping_rounds=20,
                use_best_model=True,
                thread_count=-1,
            )
            cb_trial.fit(
                Pool(X_train_sub, label=y_train_sub, cat_features=cat_feature_indices),
                eval_set=Pool(X_val_sub, label=y_val_sub, cat_features=cat_feature_indices),
                verbose=False
            )
            r2 = r2_score(np.array(y_val_sub), np.array(cb_trial.predict(X_val_sub)))

            if r2 > best_score:
                best_score        = r2
                best_params_found = sampled.copy()

        except Exception as e:
            n_failed += 1
            if n_failed ==1:
                print(f" ⚠️  Trial 0 failed (suppressing further): {e}")
            continue

        if (i + 1) % 50 == 0:
                print(f"  Iteration {i+1}/{N_ITER} — best holdout R²: {best_score:.4f}  "
                      f"(failed trials: {n_failed})")

    if best_params_found is None:
        print(f"  ⚠️  All {N_ITER} trials failed! Falling back to baseline params.")
        best_params_found = {
            'iterations':          cb_baseline.best_iteration_ or 200,
            'learning_rate':       0.05,
            'depth':               6,
            'l2_leaf_reg':         3.0,
            'random_strength':     1.0,
            'bagging_temperature': 0.5,
            'border_count':        128,
            'min_data_in_leaf':    5,
            # [FIX] 'subsample' removed — invalid with CatBoost's default
            # Bayesian bootstrap_type and would raise at construction time.
        }
        best_score = baseline_metrics['Test_R2']

    search_time          = (datetime.now() - start_time).total_seconds()
    search_times[target] = search_time

    print(f"  ✓ Search completed in {search_time:.1f}s ({search_time/60:.2f} min)")
    print(f"  Best holdout R²: {best_score:.4f}")
    print(f"  Best depth: {best_params_found.get('depth', 'N/A')}")
    print(f"  Best lr:    {best_params_found.get('learning_rate', 'N/A'):.4f}")

    # Retrain best params on full training set with early stopping
    cb_tuned = CatBoostRegressor(
        **best_params_found,
        random_seed=RANDOM_STATE,
        verbose=0,
        eval_metric='RMSE',
        use_best_model=True,
        early_stopping_rounds=30,
        cat_features=cat_feature_indices
    )
    cb_tuned.fit(es_train_pool, eval_set=es_val_pool, verbose=False)

    tuned_metrics = evaluate_model(cb_tuned, X_train, X_test, y_train, y_test)

    # CV for tuned model
    r2_mean, r2_std = catboost_cross_val_r2(best_params_found, X_train, y_train, CV_FOLDS)
    tuned_metrics['CV_R2_Mean'] = r2_mean
    tuned_metrics['CV_R2_Std']  = r2_std
    tuned_results[target] = tuned_metrics
    best_params[target]   = best_params_found

    print(f"  ✓ Tuned model:")
    print(f"    Train R²:  {tuned_metrics['Train_R2']:.4f}")
    print(f"    Test R²:   {tuned_metrics['Test_R2']:.4f}")
    print(f"    Gap:       {tuned_metrics['Gap']:.4f}")

    # ========================================================================
    # STEP 3 — SAFEGUARD
    # ========================================================================
    print("\n🛡️  Step 3: Comparing baseline vs tuned (SAFEGUARD)...")

    improvement   = tuned_metrics['Test_R2'] - baseline_metrics['Test_R2']
    gap_reduction = baseline_metrics['Gap']  - tuned_metrics['Gap']

    print(f"\n  Baseline Test R²:  {baseline_metrics['Test_R2']:.6f}")
    print(f"  Tuned Test R²:     {tuned_metrics['Test_R2']:.6f}")
    print(f"  Improvement:       {improvement:+.6f} "
          f"({100*improvement/max(baseline_metrics['Test_R2'], 0.01):+.2f}%)")
    print(f"  Gap Reduction:     {gap_reduction:+.6f}")

    if improvement > 0.001 and gap_reduction > -0.01:
        decision    = 'tuned';    final_model = cb_tuned;    final_mets = tuned_metrics
        print(f"\n  ✅ DECISION: Using TUNED model")
        print(f"      Improvement: {improvement:+.4f}, Gap reduction: {gap_reduction:+.4f}")
    elif tuned_metrics['Gap'] < baseline_metrics['Gap'] - 0.02:
        decision    = 'tuned';    final_model = cb_tuned;    final_mets = tuned_metrics
        print(f"\n  ✅ DECISION: Using TUNED model")
        print(f"      Gap reduced: {gap_reduction:.4f}")
    else:
        decision    = 'baseline'; final_model = cb_baseline; final_mets = baseline_metrics
        if improvement < -0.001:
            print(f"\n  ⚠️  DECISION: Using BASELINE (tuning DEGRADED performance)")
        else:
            print(f"\n  ℹ️  DECISION: Using BASELINE (no meaningful improvement)")

    tuning_decisions[target] = decision
    final_models[target]     = final_model
    final_results[target]    = final_mets

    # ========================================================================
    # STEP 4 — VISUALIZATIONS
    # ========================================================================
    print("\n📊 Step 4: Generating visualizations...")

    y_train_pred = final_model.predict(X_train)
    y_test_pred  = final_model.predict(X_test)

    plot_predictions(y_test, y_train, y_train_pred, y_test_pred,
                     target, f'outputs/catboost/figures/{target}_predictions.png')

    plot_learning_curves(final_model, X_train, y_train, CV_FOLDS,
                         target, f'outputs/catboost/figures/{target}_learning_curves.png')

    plot_feature_importance(final_model, X_train.columns, target,
                            f'outputs/catboost/figures/{target}_feature_importance.png')
    print("    ✓ Parity, learning curve, and feature importance plots saved")

    # ========================================================================
    # STEP 5 — SHAP ANALYSIS
    # ========================================================================
    print("\n🔍 Step 5: SHAP analysis...")

    shap_target_dir = os.path.join('outputs/catboost/figures/shap', target)
    os.makedirs(shap_target_dir, exist_ok=True)

    explainer, shap_vals, shap_df = compute_shap_values(
        final_model, X_test, X_test.columns.tolist()
    )

    plot_shap_summary(shap_vals, X_test, target,
                      save_path=os.path.join(shap_target_dir, 'shap_beeswarm.png'),
                      plot_type='dot')
    print(f"    ✓ Beeswarm plot saved")

    plot_shap_summary(shap_vals, X_test, target,
                      save_path=os.path.join(shap_target_dir, 'shap_bar.png'),
                      plot_type='bar')
    print(f"    ✓ Bar importance plot saved")

    plot_shap_dependence(shap_vals, X_test, target,
                         save_dir=shap_target_dir, top_n=5)
    print(f"    ✓ Dependence plots saved (top 5 features)")

    plot_shap_waterfall_mean(explainer, shap_vals, X_test, target,
                             save_path=os.path.join(shap_target_dir, 'shap_waterfall_mean.png'))
    print(f"    ✓ Waterfall (mean) plot saved")

    mean_abs_shap = pd.Series(np.abs(shap_vals).mean(axis=0),
                               index=X_test.columns.tolist())
    shap_importance_dict[target] = mean_abs_shap

    shap_df.to_csv(os.path.join(shap_target_dir, 'shap_values.csv'), index=False)
    mean_abs_shap.sort_values(ascending=False).to_csv(
        os.path.join(shap_target_dir, 'shap_importance.csv'), header=['mean_abs_shap']
    )
    print(f"    ✓ SHAP CSVs saved")

    # ========================================================================
    # STEP 6 — SAVE OUTPUTS
    # ========================================================================
    print("\n💾 Step 6: Saving outputs...")

    with open(f'outputs/catboost/models/{target}_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)

    train_pred_df = pd.DataFrame({
        'Split': 'Train',
        'Actual': y_train.values,
        'Predicted': y_train_pred,
        'Residual': y_train.values - y_train_pred,
        'Abs_Error': np.abs(y_train.values - y_train_pred),
        'Pct_Error': np.abs((y_train.values - y_train_pred) / (y_train.values + 1e-8)) * 100
    })
    test_pred_df = pd.DataFrame({
        'Split': 'Test',
        'Actual': y_test.values,
        'Predicted': y_test_pred,
        'Residual': y_test.values - y_test_pred,
        'Abs_Error': np.abs(y_test.values - y_test_pred),
        'Pct_Error': np.abs((y_test.values - y_test_pred) / (y_test.values + 1e-8)) * 100
    })
    pred_df = pd.concat([train_pred_df, test_pred_df], ignore_index=True)
    pred_df.to_csv(f'outputs/catboost/predictions/{target}_predictions.csv', index=False)
    
    print(f"    ✓ Model and predictions saved")


# ============================================================================
# CROSS-TARGET SHAP COMPARISON
# ============================================================================

print("\n" + "="*80)
print("CREATING CROSS-TARGET SHAP COMPARISON")
print("="*80)

plot_shap_cross_target(
    shap_importance_dict, list(TARGET_VARS),
    save_path='outputs/catboost/figures/shap/shap_cross_target_comparison.png',
    top_n=15
)
print("  ✓ Cross-target SHAP comparison plot saved")


# ============================================================================
# SUMMARY VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING SUMMARY VISUALIZATIONS")
print("="*80)

targets_list = list(TARGET_VARS)
baseline_r2  = [baseline_results[t]['Test_R2'] for t in targets_list]
tuned_r2     = [tuned_results[t]['Test_R2']    for t in targets_list]
final_r2     = [final_results[t]['Test_R2']    for t in targets_list]
x            = np.arange(len(targets_list))
width        = 0.25

fig, axes = plt.subplots(3, 1, figsize=(12, 14))

axes[0].bar(x - width, baseline_r2, width, label='Baseline',
            alpha=0.8, color='lightblue',  edgecolor='black', linewidth=0.7)
axes[0].bar(x,         tuned_r2,   width, label='Tuned',
            alpha=0.8, color='lightcoral', edgecolor='black', linewidth=0.7)
axes[0].bar(x + width, final_r2,   width, label='Final',
            alpha=0.8, color='lightgreen', edgecolor='black', linewidth=0.7)
axes[0].set_ylabel('Test R²', fontsize=12, fontweight='bold')
axes[0].set_title('CatBoost: Baseline vs Tuned vs Final', fontsize=13, fontweight='bold')
axes[0].set_xticks(x); axes[0].set_xticklabels(targets_list, rotation=45, ha='right')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].axhline(y=0.8, color='green', linestyle='--', linewidth=1, alpha=0.5)

cv_r2_mean = [final_results[t]['CV_R2_Mean'] for t in targets_list]
cv_r2_std  = [final_results[t]['CV_R2_Std']  for t in targets_list]
axes[1].bar(x, final_r2, alpha=0.7, color='steelblue',
            label='Final Test R²', edgecolor='black', linewidth=0.7)
axes[1].errorbar(x, cv_r2_mean, yerr=cv_r2_std, fmt='o',
                 color='red', capsize=5, capthick=2, markersize=8,
                 label='CV R² (mean ± std)')
axes[1].set_ylabel('R²', fontsize=12, fontweight='bold')
axes[1].set_title('Final Model Performance', fontsize=13, fontweight='bold')
axes[1].set_xticks(x); axes[1].set_xticklabels(targets_list, rotation=45, ha='right')
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

improvements = [(tuned_r2[i] - baseline_r2[i]) / max(baseline_r2[i], 0.01) * 100
                for i in range(len(targets_list))]
colors = ['green' if imp > 0 else 'red' for imp in improvements]
axes[2].bar(x, improvements, alpha=0.8, color=colors, edgecolor='black', linewidth=0.7)
axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1.5)
axes[2].set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
axes[2].set_title('Tuning Impact', fontsize=13, fontweight='bold')
axes[2].set_xticks(x); axes[2].set_xticklabels(targets_list, rotation=45, ha='right')
axes[2].grid(axis='y', alpha=0.3, linestyle='--')
for i, (imp, col) in enumerate(zip(improvements, colors)):
    axes[2].text(i, imp, f'{imp:+.1f}%', ha='center',
                 va='bottom' if imp > 0 else 'top', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/catboost/figures/performance_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Performance comparison plot saved")

fig, ax = plt.subplots(figsize=(10, 6))
search_times_list = [search_times[t] / 60 for t in targets_list]
bars = ax.barh(targets_list, search_times_list, alpha=0.8,
               color='darkorange', edgecolor='black', linewidth=0.7)
ax.set_xlabel('Search Time (minutes)', fontsize=12, fontweight='bold')
ax.set_title('RandomizedSearchCV Duration by Target', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')
for bar, v in zip(bars, search_times_list):
    ax.text(v, bar.get_y() + bar.get_height()/2, f' {v:.2f} min',
            va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/catboost/figures/search_times.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ Search time visualization saved")


# ============================================================================
# UNIFIED SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("CREATING UNIFIED SUMMARY REPORT")
print("="*80)

n_tuned           = sum(1 for t in TARGET_VARS if tuning_decisions[t] == 'tuned')
n_baseline        = len(TARGET_VARS) - n_tuned
avg_test_r2       = np.mean([final_results[t]['Test_R2'] for t in TARGET_VARS])
avg_gap           = np.mean([final_results[t]['Gap']      for t in TARGET_VARS])
total_search_time = sum(search_times.values())

summary = []
summary.append("="*80)
summary.append("CATBOOST MODEL - CATALYST PERFORMANCE PREDICTION")
summary.append("="*80)
summary.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary.append(f"Random State: {RANDOM_STATE}  |  CV Folds: {CV_FOLDS}")
summary.append(f"\nDATASET:")
summary.append(f"  Training samples: {len(X_train)}")
summary.append(f"  Test samples:     {len(X_test)}")
summary.append(f"  Features:         {len(feature_names)}")
summary.append(f"\nTARGET VARIABLES:")
for target in TARGET_VARS:
    summary.append(f"  - {target}")

summary.append(f"\n" + "="*80)
summary.append("MODEL PERFORMANCE COMPARISON")
summary.append("="*80)

for target in TARGET_VARS:
    bm = baseline_results[target]
    tm = tuned_results[target]
    fm = final_results[target]
    decision = tuning_decisions[target]
    summary.append(f"\n{target}:")
    for label, m in [("BASELINE", bm), ("TUNED", tm)]:
        mape_str = f"{m['Test_MAPE']:.2f}%" if not np.isnan(m['Test_MAPE']) else "N/A"
        summary.append(f"  {label}:  Train R²={m['Train_R2']:.4f}  "
                        f"Test R²={m['Test_R2']:.4f}  Gap={m['Gap']:.4f}  "
                        f"CV={m['CV_R2_Mean']:.4f}±{m['CV_R2_Std']:.4f}  MAPE={mape_str}")
    icon = "✅" if decision == 'tuned' else "⚠️"
    summary.append(f"  DECISION: {icon} {decision.upper()}  "
                   f"(ΔTest R²={tm['Test_R2']-bm['Test_R2']:+.4f})")
    gap = fm['Gap']
    if   gap < 0.05: summary.append(f"  Overfitting: ✓ EXCELLENT (gap={gap:.4f})")
    elif gap < 0.10: summary.append(f"  Overfitting: ✓ GOOD (gap={gap:.4f})")
    elif gap < 0.15: summary.append(f"  Overfitting: ⚠️  MODERATE (gap={gap:.4f})")
    else:            summary.append(f"  Overfitting: ⚠️  HIGH (gap={gap:.4f})")

summary.append("\n" + "="*80)
summary.append("SHAP FEATURE IMPORTANCE (TOP 10 PER TARGET)")
summary.append("="*80)
for target in TARGET_VARS:
    imp = shap_importance_dict[target].sort_values(ascending=False).head(10)
    summary.append(f"\n{target}:")
    for rank, (feat, val) in enumerate(imp.items(), 1):
        summary.append(f"  {rank:2d}. {feat:45s} {val:.6f}")

summary.append("\n" + "="*80)
summary.append("OVERALL ASSESSMENT")
summary.append("="*80)
summary.append(f"\nAverage Test R²:        {avg_test_r2:.4f}")
summary.append(f"Average Train-Test Gap: {avg_gap:.4f}")
summary.append(f"Tuned / Baseline:       {n_tuned} / {n_baseline}")
summary.append(f"Total search time:      {total_search_time:.1f}s ({total_search_time/60:.2f} min)")

if   avg_test_r2 > 0.80: summary.append("\n✓✓ EXCELLENT (R² > 0.80)")
elif avg_test_r2 > 0.70: summary.append("\n✓ GOOD (R² > 0.70)")
else:                     summary.append("\n≈ ACCEPTABLE")

if   avg_gap < 0.10: summary.append("✓ LOW overfitting (gap < 0.10)")
elif avg_gap < 0.15: summary.append("≈ MODERATE overfitting (gap 0.10-0.15)")
else:                summary.append("⚠️  HIGH overfitting (gap > 0.15)")

summary.append("\n" + "="*80)

report_text = "\n".join(summary)
with open('outputs/catboost/results/training_summary.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)
print("  ✓ Summary report saved")

training_results = {
    'metadata': {
        'generated':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model':           'CatBoost',
        'random_state':    int(RANDOM_STATE),
        'cv_folds':        int(CV_FOLDS),
        'n_features':      len(feature_names),
        'n_train_samples': len(X_train),
        'n_test_samples':  len(X_test),
    },
    'baseline_results': {
        t: {k: (float(v) if isinstance(v, (np.floating, np.integer)) and not np.isnan(v)
                else (None if isinstance(v, float) and np.isnan(v) else v))
            for k, v in baseline_results[t].items()} for t in TARGET_VARS},
    'tuned_results': {
        t: {k: (float(v) if isinstance(v, (np.floating, np.integer)) and not np.isnan(v)
                else (None if isinstance(v, float) and np.isnan(v) else v))
            for k, v in tuned_results[t].items()} for t in TARGET_VARS},
    'final_results': {
        t: {k: (float(v) if isinstance(v, (np.floating, np.integer)) and not np.isnan(v)
                else (None if isinstance(v, float) and np.isnan(v) else v))
            for k, v in final_results[t].items()} for t in TARGET_VARS},
    'tuning_decisions': tuning_decisions,
    'search_times':     {t: float(search_times[t]) for t in TARGET_VARS},
    'shap_top10': {
        t: shap_importance_dict[t].sort_values(ascending=False).head(10).to_dict()
        for t in TARGET_VARS
    }
}

with open('outputs/catboost/results/training_results.json', 'w') as f:
    json.dump(training_results, f, indent=2)
print("  ✓ JSON results saved")


# ============================================================================
# FINAL CONSOLE OUTPUT
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

for target in TARGET_VARS:
    final         = final_results[target]
    decision      = tuning_decisions[target]
    decision_icon = "✅" if decision == 'tuned' else "⚠️"
    gap_status    = "✓" if final['Gap'] < 0.10 else "⚠️"
    print(f"\n{target} ({decision_icon} {decision.upper()}):")
    print(f"  Test R²:  {final['Test_R2']:.4f}")
    print(f"  CV R²:    {final['CV_R2_Mean']:.4f} (±{final['CV_R2_Std']:.4f})")
    print(f"  Gap:      {final['Gap']:.4f} {gap_status}")

print(f"\n{'='*80}")
print(f"Average Test R²: {avg_test_r2:.4f}  |  Average Gap: {avg_gap:.4f}")
print(f"Tuned: {n_tuned}/{len(TARGET_VARS)}  |  "
      f"Total search time: {total_search_time:.1f}s ({total_search_time/60:.2f} min)")
print("\n📁 Outputs: outputs/catboost/")
print("\n✅ CatBoost production training complete!")
print("="*80)
