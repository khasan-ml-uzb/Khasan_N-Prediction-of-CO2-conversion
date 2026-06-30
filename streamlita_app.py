"""
streamlita_app.py
=================================================================================
Streamlit app — Optimal Catalyst Recommender for CO2 Hydrogenation to Gasoline
=================================================================================

This app deploys the trained XGBoost surrogate models (created by ``XGBoost.py``)
and the Bayesian-Optimization-style catalyst search (from ``Bayesian_optimization.py``)
into an interactive web tool.

Workflow
--------
1.  ``XGBoost.py``  trains three surrogate models and pickles them:
        - CO2_conv      → CO2 conversion (%)
        - C5plus_sel    → C5+ selectivity (%)
        - C5plus_yield  → C5+ yield (%)
2.  ``Bayesian_optimization.py`` searches the Fe/K/Al + Me1/Me2 composition space
    to recommend Pareto-optimal catalysts.
3.  This app lets the user change the three operating conditions
        - Temperature (°C)
        - Pressure (MPa)
        - GHSV (mL/g/h)
    and, for those conditions, recommends the **optimal catalyst composition**
    (Fe/K/Al + two selectable promoters) by maximising the chosen objective with
    a fast vectorised random search over the surrogate models. A manual
    single-point prediction mode is also provided.

Deploy on Streamlit Cloud
-------------------------
    - Commit the trained ``*_model.pkl`` files into the repo (see model search
      paths below) together with this file and ``requirements.txt``.
    - Point Streamlit Cloud at ``streamlita_app.py``.
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CO₂→Gasoline Catalyst Optimizer",
    page_icon="⚗️",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION  (ported from Bayesian_optimization.py so the app and the
#     offline optimiser describe an identical search space)
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Target → list of candidate filenames to search for the pickled model.
MODEL_FILES = {
    "CO2_conv":     ["CO2_conv_model.pkl"],
    "C5plus_sel":   ["C5plus_sel_model.pkl"],
    "C5plus_yield": ["C5plus_yield_model.pkl"],
}

# Directories that may hold the pickled models, in priority order.
MODEL_DIRS = [
    _SCRIPT_DIR,
    os.path.join(_SCRIPT_DIR, "models"),
    os.path.join(_SCRIPT_DIR, "outputs", "xgboost_v3", "models"),
    os.path.join(_SCRIPT_DIR, "outputs", "models"),
]

# ── Default fixed synthesis conditions (encoder integer codes) ────────────────
# These mirror the Bayesian_optimization.py defaults. The label-encoder mapping
# lives inside XGBoost.py at train time; the integer codes below match the
# values used by the offline optimiser. Adjust in the sidebar if your encoder
# order differs.
DEFAULT_SUPPORT_CODE = 3       # 3 = bulk (unsupported); 1 = Carbon support
DEFAULT_SYNTH_CODE   = 3       # 3 = precipitation (verify your encoder order)
DEFAULT_CALC_TEMP_C  = 400.0   # °C
DEFAULT_REDUC_TEMP_C = 380.0   # °C
DEFAULT_H2_CO2       = 3.0     # H2/CO2 molar ratio

# ── Composition search space (Fe/K/Al fixed roles, Me1/Me2 selectable) ────────
FE_WT_RANGE = (60.0, 92.0)     # wt%  active metal
K_WT_RANGE  = (1.0, 25.0)      # wt%  alkali promoter
AL_WT_RANGE = (3.0, 25.0)      # wt%  structural promoter
ME_WT_RANGE = (0.5, 12.0)      # wt%  additional promoters

# Promoter pool for Me1 / Me2 (Fe, K, Al are fixed roles and excluded here).
ME_POOL = {
    "Cu": "Cu_wt",
    "Na": "Na_wt",
    "Zn": "Zn_wt",
    "Mn": "Mn_wt",
    "Co": "Co_wt",
    "Zr": "Zr_wt",
}

# ── Operating-condition ranges (slider bounds) ────────────────────────────────
TEMP_C_RANGE = (300.0, 360.0)   # °C
P_MPA_RANGE  = (1.5, 3.0)       # MPa
GHSV_RANGE   = (2000.0, 9000.0) # mL/g/h

# ══════════════════════════════════════════════════════════════════════════════
# 2.  MENDELEEV DESCRIPTORS  (ported from Bayesian_optimization.py)
# ══════════════════════════════════════════════════════════════════════════════
EN = {
    "Fe": 1.83, "K": 0.82, "Na": 0.93, "Al": 1.61,
    "Cu": 1.90, "Co": 1.88, "Zn": 1.65, "Mn": 1.55, "Zr": 1.33,
}
COL_TO_ELEM = {
    "Fe_wt": "Fe", "K_wt": "K", "Na_wt": "Na", "Al_wt": "Al",
    "Cu_wt": "Cu", "Co_wt": "Co", "Zn_wt": "Zn", "Mn_wt": "Mn", "Zr_wt": "Zr",
}
D_ELECTRONS = {
    "Fe": 6, "K": 0, "Na": 0, "Al": 0,
    "Cu": 9, "Co": 7, "Zn": 10, "Mn": 5, "Zr": 2,
}
ELEM_COLS = list(COL_TO_ELEM.keys())   # canonical composition-weight columns


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _find_model_path(filenames):
    """Return the first existing path for any of ``filenames`` across MODEL_DIRS."""
    for d in MODEL_DIRS:
        for fn in filenames:
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                return p
    # Last resort: recursive glob under the script directory.
    for fn in filenames:
        hits = glob.glob(os.path.join(_SCRIPT_DIR, "**", fn), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


@st.cache_resource(show_spinner=False)
def load_models():
    """Load whichever target models are available. Returns (models, feature_cols, errors)."""
    models, errors = {}, {}
    for target, filenames in MODEL_FILES.items():
        path = _find_model_path(filenames)
        if path is None:
            errors[target] = "not found"
            continue
        try:
            with open(path, "rb") as f:
                models[target] = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            errors[target] = f"{type(exc).__name__}: {exc}"

    feature_cols = None
    for mdl in models.values():
        try:
            feature_cols = list(mdl.get_booster().feature_names)
        except Exception:  # noqa: BLE001
            feature_cols = None
        if feature_cols:
            break
    return models, feature_cols, errors


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FEATURE ENGINEERING  (vectorised replica of build_feature_matrix)
# ══════════════════════════════════════════════════════════════════════════════
def _descriptors(weights):
    """Vectorised d_w and EN_FeP for a (n, n_elem) weight matrix (already in wt%).

    ``weights`` columns follow ELEM_COLS order. EN_FeP excludes the alkali (K).
    """
    n_elem = weights.shape[1]
    d_vec = np.array([D_ELECTRONS[COL_TO_ELEM[c]] for c in ELEM_COLS])
    en_vec = np.array([EN[COL_TO_ELEM[c]] for c in ELEM_COLS])
    k_idx = ELEM_COLS.index("K_wt")

    total = weights.sum(axis=1)
    total_safe = np.where(total > 0, total, 1.0)
    d_w = (weights * d_vec).sum(axis=1) / total_safe

    non_alk = weights.copy()
    non_alk[:, k_idx] = 0.0
    total_na = non_alk.sum(axis=1)
    total_na_safe = np.where(total_na > 0, total_na, 1.0)
    en_fep = (non_alk * en_vec).sum(axis=1) / total_na_safe
    return d_w, en_fep


def build_feature_frame(weights, temp_c, p_mpa, ghsv, fixed, feature_cols):
    """Build a model-ready DataFrame.

    Parameters
    ----------
    weights : (n, n_elem) array of composition weights in wt% (already normalised).
    temp_c, p_mpa, ghsv : scalars or (n,) arrays of operating conditions.
    fixed : dict with support, synth, calc_temp_C, reduc_temp_C, H2_CO2.
    feature_cols : ordered list of feature names the model expects.
    """
    n = weights.shape[0]
    d_w, en_fep = _descriptors(weights)

    data = {col: np.zeros(n) for col in feature_cols}

    def _set(col, val):
        if col in data:
            data[col] = np.full(n, val) if np.isscalar(val) else np.asarray(val)

    # Fixed synthesis conditions
    _set("support", fixed["support"])
    _set("synth", fixed["synth"])
    _set("calc_temp_C", fixed["calc_temp_C"])
    _set("reduc_temp_C", fixed["reduc_temp_C"])
    _set("H2_CO2", fixed["H2_CO2"])

    # Operating conditions
    _set("temp_C", temp_c)
    _set("P_MPa", p_mpa)
    _set("GHSV", ghsv)

    # Descriptors
    _set("d_w", d_w)
    _set("EN_FeP", en_fep)

    # Composition weights
    for j, col in enumerate(ELEM_COLS):
        _set(col, weights[:, j])

    return pd.DataFrame(data, columns=feature_cols)


def _normalise(weights):
    """Row-normalise a weight matrix to sum to 100 wt%."""
    total = weights.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    return weights / total * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def predict_targets(models, frame):
    """Predict every available target; return dict target -> (n,) array, clipped 0..100."""
    out = {}
    for target, mdl in models.items():
        try:
            pred = np.asarray(mdl.predict(frame), dtype=float)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed for {target}: {exc}")
            continue
        # C5plus_yield model was trained in log1p space → invert.
        if target == "C5plus_yield":
            pred = np.expm1(pred)
        out[target] = np.clip(pred, 0.0, 100.0)
    return out


def objective_values(preds, objective):
    """Compute the scalar objective to maximise from a prediction dict."""
    co2 = preds.get("CO2_conv")
    sel = preds.get("C5plus_sel")
    yld = preds.get("C5plus_yield")
    if objective == "C5+ yield" and yld is not None:
        return yld
    if objective == "C5+ selectivity" and sel is not None:
        return sel
    if objective == "CO2 conversion" and co2 is not None:
        return co2
    # Composite = mean of available CO2 conversion and C5+ yield.
    parts = [p for p in (co2, yld) if p is not None]
    if not parts:
        parts = [p for p in preds.values()]
    return np.mean(parts, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  OPTIMAL-CATALYST SEARCH  (fast vectorised random search)
# ══════════════════════════════════════════════════════════════════════════════
def sample_compositions(n, me_elements, rng):
    """Sample n random Fe/K/Al + Me1/Me2 catalysts → normalised weight matrix."""
    n_me = len(me_elements)
    me_col_idx = [ELEM_COLS.index(ME_POOL[e]) for e in me_elements]

    fe = rng.uniform(*FE_WT_RANGE, n)
    k = rng.uniform(*K_WT_RANGE, n)
    al = rng.uniform(*AL_WT_RANGE, n)
    me1_sel = rng.integers(0, n_me, n)
    me2_sel = rng.integers(0, n_me, n)
    # Force Me1 != Me2
    clash = me1_sel == me2_sel
    me2_sel[clash] = (me2_sel[clash] + 1) % n_me
    me1_wt = rng.uniform(*ME_WT_RANGE, n)
    me2_wt = rng.uniform(*ME_WT_RANGE, n)

    weights = np.zeros((n, len(ELEM_COLS)))
    weights[:, ELEM_COLS.index("Fe_wt")] = fe
    weights[:, ELEM_COLS.index("K_wt")] = k
    weights[:, ELEM_COLS.index("Al_wt")] = al
    rows = np.arange(n)
    weights[rows, np.array(me_col_idx)[me1_sel]] += me1_wt
    weights[rows, np.array(me_col_idx)[me2_sel]] += me2_wt

    weights = _normalise(weights)
    me1_elem = [me_elements[i] for i in me1_sel]
    me2_elem = [me_elements[i] for i in me2_sel]
    return weights, me1_elem, me2_elem


def optimise_catalyst(models, temp_c, p_mpa, ghsv, fixed, feature_cols,
                      me_elements, objective, n_samples, top_n, seed):
    """Random-search the composition space at fixed operating conditions."""
    rng = np.random.default_rng(seed)
    weights, me1_elem, me2_elem = sample_compositions(n_samples, me_elements, rng)
    frame = build_feature_frame(weights, temp_c, p_mpa, ghsv, fixed, feature_cols)
    preds = predict_targets(models, frame)
    if not preds:
        return None
    score = objective_values(preds, objective)

    order = np.argsort(score)[::-1][:top_n]
    records = []
    for rank, i in enumerate(order, 1):
        rec = {"Rank": rank, "Objective": round(float(score[i]), 3)}
        rec["Fe_wt%"] = round(float(weights[i, ELEM_COLS.index("Fe_wt")]), 2)
        rec["K_wt%"] = round(float(weights[i, ELEM_COLS.index("K_wt")]), 2)
        rec["Al_wt%"] = round(float(weights[i, ELEM_COLS.index("Al_wt")]), 2)
        rec["Me1"] = me1_elem[i]
        rec["Me1_wt%"] = round(float(weights[i, ELEM_COLS.index(ME_POOL[me1_elem[i]])]), 2)
        rec["Me2"] = me2_elem[i]
        rec["Me2_wt%"] = round(float(weights[i, ELEM_COLS.index(ME_POOL[me2_elem[i]])]), 2)
        for target, label in (("CO2_conv", "CO2_conv_%"),
                              ("C5plus_sel", "C5+_sel_%"),
                              ("C5plus_yield", "C5+_yield_%")):
            if target in preds:
                rec[label] = round(float(preds[target][i]), 2)
        records.append(rec)
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SIDEBAR  — operating conditions & settings
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("⚗️ Settings")

st.sidebar.header("Operating conditions")
temp_c = st.sidebar.slider(
    "Temperature (°C)", 250.0, 400.0, 330.0, 1.0,
    help="Reaction temperature. Data peak ≈ 320–340 °C.",
)
p_mpa = st.sidebar.slider(
    "Pressure (MPa)", 1.0, 3.5, 2.25, 0.05,
    help="Reaction pressure. Data-supported optimum ≈ 2.0–2.5 MPa.",
)
ghsv = st.sidebar.slider(
    "GHSV (mL/g/h)", 1000.0, 10000.0, 5000.0, 100.0,
    help="Gas hourly space velocity.",
)

st.sidebar.header("Objective")
objective = st.sidebar.selectbox(
    "Maximise",
    ["C5+ yield", "Composite (CO2 conv & C5+ yield)", "C5+ selectivity", "CO2 conversion"],
    index=0,
)

st.sidebar.header("Promoter pool (Me1 / Me2)")
me_elements = st.sidebar.multiselect(
    "Candidate promoters",
    list(ME_POOL.keys()),
    default=["Cu", "Na", "Zn", "Mn"],
    help="Elements the optimiser may choose for the two additional promoters.",
)

with st.sidebar.expander("Fixed synthesis conditions"):
    support_code = st.number_input("Support code", value=DEFAULT_SUPPORT_CODE, step=1,
                                   help="Label-encoded support (e.g. 3 = bulk).")
    synth_code = st.number_input("Synthesis code", value=DEFAULT_SYNTH_CODE, step=1,
                                 help="Label-encoded synthesis route (e.g. 3 = precipitation).")
    calc_temp = st.number_input("Calcination temp (°C)", value=DEFAULT_CALC_TEMP_C, step=10.0)
    reduc_temp = st.number_input("Reduction temp (°C)", value=DEFAULT_REDUC_TEMP_C, step=10.0)
    h2_co2 = st.number_input("H₂/CO₂ ratio", value=DEFAULT_H2_CO2, step=0.5)

with st.sidebar.expander("Search settings"):
    n_samples = st.select_slider(
        "Random samples", options=[2000, 5000, 10000, 20000, 50000], value=10000,
    )
    top_n = st.slider("Show top N catalysts", 1, 50, 10)
    seed = st.number_input("Random seed", value=42, step=1)

FIXED = {
    "support": int(support_code),
    "synth": int(synth_code),
    "calc_temp_C": float(calc_temp),
    "reduc_temp_C": float(reduc_temp),
    "H2_CO2": float(h2_co2),
}

# ══════════════════════════════════════════════════════════════════════════════
# 8.  HEADER & MODEL STATUS
# ══════════════════════════════════════════════════════════════════════════════
st.title("CO₂ Hydrogenation — Optimal Catalyst Recommender")
st.caption(
    "Fe–K–Al + Me1/Me2 iron catalysts for CO₂-to-gasoline. XGBoost surrogate "
    "models recommend the optimal catalyst at your chosen Temperature, Pressure "
    "and GHSV."
)

models, feature_cols, errors = load_models()

if not models or feature_cols is None:
    st.error("⚠️  No trained XGBoost model files were found.")
    st.markdown(
        f"""
The app looks for these pickled models (any one is enough to start, all three
unlock full predictions):

| Target | Filename |
|---|---|
| CO₂ conversion | `CO2_conv_model.pkl` |
| C5+ selectivity | `C5plus_sel_model.pkl` |
| C5+ yield | `C5plus_yield_model.pkl` |

**Search locations** (relative to this app):
{os.linesep.join('- `' + os.path.relpath(d, _SCRIPT_DIR) + '`' for d in MODEL_DIRS)}

**To generate them:** run `python XGBoost.py` on your dataset, then copy the
`*_model.pkl` files into one of the locations above (or commit them next to this
app for Streamlit Cloud).
"""
    )
    if errors:
        with st.expander("Loader details"):
            st.json(errors)
    st.stop()

loaded = ", ".join(sorted(models.keys()))
st.success(f"Loaded models: **{loaded}**  ·  {len(feature_cols)} features")
if errors:
    st.info("Missing models (predictions for these targets are unavailable): "
            + ", ".join(sorted(errors.keys())))

# ══════════════════════════════════════════════════════════════════════════════
# 9.  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_opt, tab_manual, tab_sweep, tab_about = st.tabs(
    ["🎯 Optimal catalyst", "🧪 Manual prediction", "📈 Condition sweep", "ℹ️ About"]
)

# ── TAB 1: Optimal catalyst recommendation ────────────────────────────────────
with tab_opt:
    st.subheader("Recommended optimal catalyst for the selected conditions")
    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature", f"{temp_c:.0f} °C")
    c2.metric("Pressure", f"{p_mpa:.2f} MPa")
    c3.metric("GHSV", f"{ghsv:.0f} mL/g/h")

    if not me_elements or len(me_elements) < 2:
        st.warning("Select at least two promoter elements in the sidebar.")
    elif st.button("🔍 Find optimal catalyst", type="primary"):
        with st.spinner(f"Searching {n_samples:,} candidate compositions…"):
            result = optimise_catalyst(
                models, temp_c, p_mpa, ghsv, FIXED, feature_cols,
                me_elements, objective, int(n_samples), int(top_n), int(seed),
            )
        if result is None or result.empty:
            st.error("Optimisation produced no result — check that models loaded correctly.")
        else:
            best = result.iloc[0]
            st.markdown("### 🥇 Best catalyst")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("CO₂ conversion",
                      f"{best.get('CO2_conv_%', float('nan')):.1f} %"
                      if "CO2_conv_%" in result.columns else "—")
            b2.metric("C5+ selectivity",
                      f"{best.get('C5+_sel_%', float('nan')):.1f} %"
                      if "C5+_sel_%" in result.columns else "—")
            b3.metric("C5+ yield",
                      f"{best.get('C5+_yield_%', float('nan')):.1f} %"
                      if "C5+_yield_%" in result.columns else "—")
            b4.metric("Objective", f"{best['Objective']:.2f}")

            comp = (f"Fe {best['Fe_wt%']} · K {best['K_wt%']} · Al {best['Al_wt%']} · "
                    f"{best['Me1']} {best['Me1_wt%']} · {best['Me2']} {best['Me2_wt%']}  (wt%)")
            st.info(f"**Composition:** {comp}")

            st.markdown(f"### Top {len(result)} catalysts")
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download recommendations (CSV)",
                result.to_csv(index=False).encode("utf-8"),
                file_name=f"optimal_catalysts_T{temp_c:.0f}_P{p_mpa:.2f}_GHSV{ghsv:.0f}.csv",
                mime="text/csv",
            )
    else:
        st.info("Set the Temperature, Pressure and GHSV in the sidebar, then click "
                "**Find optimal catalyst**.")

# ── TAB 2: Manual single prediction ───────────────────────────────────────────
with tab_manual:
    st.subheader("Predict performance for a specific catalyst")
    st.caption("Enter a composition (auto-normalised to 100 wt%) and the operating "
               "conditions from the sidebar are used.")

    m1, m2, m3 = st.columns(3)
    fe_in = m1.number_input("Fe (wt%)", 0.0, 100.0, 80.0, 1.0)
    k_in = m2.number_input("K (wt%)", 0.0, 100.0, 12.0, 0.5)
    al_in = m3.number_input("Al (wt%)", 0.0, 100.0, 5.0, 0.5)

    m4, m5, m6, m7 = st.columns(4)
    me1_el = m4.selectbox("Promoter 1", list(ME_POOL.keys()), index=0)
    me1_in = m5.number_input(f"{me1_el} (wt%)", 0.0, 100.0, 2.0, 0.5)
    me2_el = m6.selectbox("Promoter 2", list(ME_POOL.keys()), index=1)
    me2_in = m7.number_input(f"{me2_el} (wt%)", 0.0, 100.0, 1.0, 0.5)

    if st.button("Predict performance"):
        w = np.zeros((1, len(ELEM_COLS)))
        w[0, ELEM_COLS.index("Fe_wt")] = fe_in
        w[0, ELEM_COLS.index("K_wt")] = k_in
        w[0, ELEM_COLS.index("Al_wt")] = al_in
        w[0, ELEM_COLS.index(ME_POOL[me1_el])] += me1_in
        w[0, ELEM_COLS.index(ME_POOL[me2_el])] += me2_in
        w = _normalise(w)
        frame = build_feature_frame(w, temp_c, p_mpa, ghsv, FIXED, feature_cols)
        preds = predict_targets(models, frame)
        if preds:
            cols = st.columns(len(preds))
            labels = {"CO2_conv": "CO₂ conversion",
                      "C5plus_sel": "C5+ selectivity",
                      "C5plus_yield": "C5+ yield"}
            for col, (target, val) in zip(cols, preds.items()):
                col.metric(labels.get(target, target), f"{float(val[0]):.2f} %")
            norm = {ELEM_COLS[j]: round(float(w[0, j]), 2)
                    for j in range(len(ELEM_COLS)) if w[0, j] > 0}
            st.caption(f"Normalised composition (wt%): {norm}")

# ── TAB 3: Sweep one operating condition ──────────────────────────────────────
with tab_sweep:
    st.subheader("How the optimum changes with one operating condition")
    st.caption("Holds the other two conditions at their sidebar values and re-optimises "
               "the catalyst at each point along the swept variable.")

    sweep_var = st.selectbox("Sweep variable", ["Temperature (°C)", "Pressure (MPa)", "GHSV (mL/g/h)"])
    n_points = st.slider("Points", 3, 25, 9)
    sweep_samples = st.select_slider("Samples per point", options=[1000, 2000, 5000, 10000], value=2000)

    if not me_elements or len(me_elements) < 2:
        st.warning("Select at least two promoter elements in the sidebar.")
    elif st.button("Run sweep"):
        ranges = {
            "Temperature (°C)": TEMP_C_RANGE,
            "Pressure (MPa)": P_MPA_RANGE,
            "GHSV (mL/g/h)": GHSV_RANGE,
        }
        grid = np.linspace(ranges[sweep_var][0], ranges[sweep_var][1], n_points)
        rows = []
        prog = st.progress(0.0)
        for k, val in enumerate(grid):
            t, p, g = temp_c, p_mpa, ghsv
            if sweep_var == "Temperature (°C)":
                t = val
            elif sweep_var == "Pressure (MPa)":
                p = val
            else:
                g = val
            res = optimise_catalyst(models, t, p, g, FIXED, feature_cols,
                                    me_elements, objective, int(sweep_samples), 1, int(seed))
            if res is not None and not res.empty:
                r = res.iloc[0].to_dict()
                r[sweep_var] = round(float(val), 2)
                rows.append(r)
            prog.progress((k + 1) / n_points)
        prog.empty()

        if rows:
            df = pd.DataFrame(rows)
            ycols = [c for c in ["CO2_conv_%", "C5+_sel_%", "C5+_yield_%"] if c in df.columns]
            st.line_chart(df.set_index(sweep_var)[ycols])
            st.dataframe(df, use_container_width=True, hide_index=True)

# ── TAB 4: About ──────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
### About this app

This tool deploys the surrogate models and optimisation strategy from the study:

> **Mitigating the trade-off between selectivity and stability in CO₂
> hydrogenation using ultra-high potassium loading on iron catalysts**
> — *Fuel* 427 (2027) 139918 · https://doi.org/10.1016/j.fuel.2026.139918

**Pipeline**
1. `XGBoost.py` trains three surrogate models (CO₂ conversion, C5+ selectivity,
   C5+ yield) and pickles them.
2. `Bayesian_optimization.py` performs ParEGO multi-objective Bayesian
   optimisation to recommend Pareto-optimal Fe–K–Al + Me1/Me2 catalysts.
3. **This app** lets you change Temperature, Pressure and GHSV and, for those
   conditions, recommends the optimal catalyst via a fast vectorised random
   search over the same surrogate models and search space.

**Catalyst architecture**
- **Fe** — active metal (60–92 wt%)
- **K** — alkali promoter (1–25 wt%)
- **Al** — structural promoter (3–25 wt%)
- **Me1 / Me2** — two additional promoters chosen from
  Cu, Na, Zn, Mn, Co, Zr (0.5–12 wt% each)

All weights are normalised to 100 wt%. The `C5plus_yield` model is trained in
`log1p` space and predictions are inverted automatically.

> The `support`/`synth` codes are the label-encoded integers produced at train
> time in `XGBoost.py`. If your encoder order differs, adjust the codes under
> *Fixed synthesis conditions* in the sidebar.
"""
    )
