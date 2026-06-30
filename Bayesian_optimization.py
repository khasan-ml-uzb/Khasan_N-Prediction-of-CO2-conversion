"""
Bayesian Optimisation — Fe/K/Al Fixed + Me1/Me2 Selectable Promoters
======================================================================
Catalyst architecture (fixed elements, optimised weights):
    Fe   — active metal          (FE_WT_RANGE)
    K    — alkali promoter       (K_WT_RANGE)       ← was K/Na binary selector
    Al   — structural promoter   (AL_WT_RANGE)       ← was inside OTHER_METALS pool
    Me1  — additional promoter 1 (ME1_WT_RANGE, element chosen by BO)
    Me2  — additional promoter 2 (ME2_WT_RANGE, element chosen by BO)
    All five sum to 100 wt% after normalisation.

Decision variables (10 total — same count as v9):
    x[0]  Fe_wt           FE_WT_RANGE        (pre-normalisation)
    x[1]  K_wt            K_WT_RANGE         (pre-normalisation)
    x[2]  Al_wt           AL_WT_RANGE        (pre-normalisation)
    x[3]  Me1 index       [0, n_me-1]        integer (real, rounded)
    x[4]  Me1_wt          ME_WT_RANGE        (pre-normalisation)
    x[5]  Me2 index       [0, n_me-1]        integer (real, rounded)
    x[6]  Me2_wt          ME_WT_RANGE        (pre-normalisation)
    x[7]  temp_C          TEMP_C_RANGE       (optimised)
    x[8]  P_MPa           P_MPA_RANGE        (optimised)
    x[9]  GHSV            GHSV_RANGE         (optimised)

Changes vs v9
-------------
  1. x[1] was K/Na binary selector → now K_wt (K is always the alkali)
  2. Al removed from ME_POOL → now fixed structural promoter (x[2])
  3. FE_WT_RANGE widened to 60–92 %   (data: IQR for yield>20% is 75–91%)
  4. K_WT_RANGE widened to 1–25 %     (data: Carbon 10–15%, Al2O3 20–26%, Cu-bulk 1–3%)
  5. AL_WT_RANGE set to 3–25 %        (data: Al promoter optimal 8–20%)
  6. TEMP_C_RANGE narrowed 300–360°C  (data peak: 320–340°C)
  7. P_MPA_RANGE narrowed 1.5–3.0 MPa (data peak: 2.0–2.5 MPa)
  8. GHSV_RANGE narrowed 2000–9000    (data peak: 3600–9000)
  9. REDUC_TEMP_C corrected 500→380°C (data optimal: 340–400°C)
 10. CALC_TEMP_C corrected 500→400°C  (data optimal: 350–500°C; 500 is max)
 11. ME_POOL excludes Al (now fixed) and K (now fixed)
"""

import os, pickle, warnings, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

try:
    from skopt import gp_minimize
    from skopt.space import Real
except ImportError:
    raise ImportError(
        "scikit-optimize is required.  Install with:\n"
        "    pip install scikit-optimize"
    )

from pymoo.indicators.hv import HV
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "CO2_conversion": os.path.join(_SCRIPT_DIR, "CO2_conv_model.pkl"),
    "C5+_yield":      os.path.join(_SCRIPT_DIR, "C5plus_yield_model.pkl"),
}

# ── Fixed synthesis conditions ────────────────────────────────────────────────
# SUPPORT_CODE: check your OrdinalEncoder order.
#   Typical sklearn OrdinalEncoder alphabetical: Al2O3=0, Carbon=1, TiO2=2, bulk=3, others=4
#   → bulk=3 (unsupported, most data), Carbon=1 (best consistent yield)
#   Recommend bulk (3) for precipitation route; Carbon (1) for impregnation route.
SUPPORT_CODE  = 3       # 3 = bulk (unsupported); change to 1 for Carbon support
SYNTH_CODE    = 3       # verify your encoder: precipitation is typically 3 or 4
                        # (alphabetical: autoclave=0, combustion_template=1,
                        #  impregnation=2, precipitation=3, sol_gel=4, ...)

# ── CORRECTED preparation temperatures (v9 had both at 500°C — too high) ─────
#    Data analysis: calc 350–500°C optimal, reduc 340–400°C optimal.
CALC_TEMP_C   = 400.0   # °C  [was 500 in v9 — corrected to centre of optimal range]
REDUC_TEMP_C  = 380.0   # °C  [was 500 in v9 — corrected; 500°C clearly sub-optimal]

H2_CO2        = 3.0     # fixed (>96% of high-yield experiments use H2/CO2=3)

# ── Composition search space — Fe/K/Al fixed, Me1/Me2 selectable ─────────────
#
# Fe: data shows IQR for yield>20% is 75–91 wt% Fe.
#     Lower bound 60% allows BO to explore mid-Fe formulations.
#     Upper bound 92% prevents pure-Fe (no promoters) compositions.
FE_WT_RANGE   = (60.0, 92.0)   # wt%  [was (40, 80) in v9 — too low and narrow]

# K: non-monotonic effect; two optimal regimes:
#     - 1–3% K (with Cu, bulk autoclave route)
#     - 10–15% K (Carbon support, impregnation)
#     - 20–26% K (Al2O3 support, impregnation — highest single yield 30.76%)
#   Wide range lets BO discover which regime fits the chosen support/synth.
K_WT_RANGE    = (1.0, 25.0)    # wt%  [NEW — v9 used ALK_WT_RANGE (1,10), too narrow]

# Al: structural promoter in precipitated bulk catalysts.
#     Data: Fe-Al-Na system mean yield 18.84%; optimal Al content 8–20%.
#     Lower bound 3% to allow trace-level structural roles.
AL_WT_RANGE   = (3.0, 25.0)    # wt%  [NEW — Al was inside OTHER_METALS in v9]

# Me1 / Me2: additional promoters chosen from ME_POOL.
#     Ranges are kept moderate; Na and Zn are trace/low-level promoters,
#     Cu and Mn can be used at higher levels.
ME_WT_RANGE   = (0.5, 12.0)    # wt%  [was (1, 20) — upper trimmed, lower relaxed]

# ── Additional promoter pool (Me1 & Me2) ─────────────────────────────────────
# Al removed (now fixed structural promoter).
# K removed (now fixed alkali promoter).
# Ordered by evidence of positive effect from dataset analysis:
#   Cu (+1.85%), Na (+2.92%), Zn (neutral), Mn (negative alone, ok with K),
#   Co (negative on average), Zr (most negative)
# All are included to give BO full freedom; negative promoters are penalised
# naturally by the surrogate model predictions.
ME_POOL = {
    "Cu": "Cu_wt",   # ★ Reduction facilitator; optimal ~7.5% with autoclave
    "Na": "Na_wt",   # ★ Electronic promoter; optimal 1–1.5% (trace amounts)
    "Zn": "Zn_wt",   # Structural/phase stabiliser; neutral effect
    "Mn": "Mn_wt",   # Structural; negative alone, ok combined with K
    "Co": "Co_wt",   # Second active metal; negative on average (use with caution)
    "Zr": "Zr_wt",   # Structural; currently negative — kept for exploration
}

# ── Narrowed operating condition ranges ──────────────────────────────────────
# All three ranges are narrowed based on the conditions analysis:
#   Temperature: bell-shaped peak at 320–340°C; outside 300–360°C drops sharply
#   Pressure: monotonic increase; 2.0–2.5 MPa is the data-supported optimum
#   GHSV: 2000–9000 covers 90% of high-yield experiments; <2000 underperforms
TEMP_C_RANGE  = (300.0, 360.0)   # °C        [was (250, 400) — too wide]
P_MPA_RANGE   = (1.5,   3.0)     # MPa       [was (1.0, 3.5) — sub-optimal tails removed]
GHSV_RANGE    = (2000.0, 9000.0) # mL/g/h    [was (1000, 10000) — low end underperforms]

# ── BO hyper-parameters (unchanged from v9) ──────────────────────────────────
N_RESTARTS            = 120
N_CALLS_PER_RESTART   = 60
N_INITIAL_PER_RESTART = 30
RHO                   = 0.05
SEED                  = 42
STABILITY_SEEDS       = [42, 100, 123, 456, 789, 1000]
HV_REF_POINT          = np.array([0.0, 0.0])
OUTPUT_DIR            = "outputs/BO_Fe_K_Al_Me1Me2_V10"
TOP_N                 = 100


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE COLUMNS  — populated dynamically from model booster
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLUMNS: list = []
N_FEATURES:      int  = 0
_COL_IDX:        dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MENDELEEV DESCRIPTORS
# ══════════════════════════════════════════════════════════════════════════════

EN = {
    "Fe": 1.83, "K":  0.82, "Na": 0.93, "Al": 1.61,
    "Cu": 1.90, "Co": 1.88, "Zn": 1.65, "Mn": 1.55,
    "Zr": 1.33,
}
COL_TO_ELEM = {
    "Fe_wt": "Fe", "K_wt":  "K",  "Na_wt": "Na", "Al_wt": "Al",
    "Cu_wt": "Cu", "Co_wt": "Co", "Zn_wt": "Zn", "Mn_wt": "Mn",
    "Zr_wt": "Zr",
}
D_ELECTRONS = {
    "Fe": 6, "K":  0, "Na": 0, "Al": 0,
    "Cu": 9, "Co": 7, "Zn": 10, "Mn": 5,
    "Zr": 2,
}


def _compute_descriptors(wt_dict: dict) -> dict:
    """
    Compute d_w and EN_FeP.
    EN_FeP always excludes K (fixed alkali). Al is now included in the
    non-alkali group (it is a structural promoter, not an alkali).
    """
    nonzero  = {col: wt for col, wt in wt_dict.items() if wt > 0}
    total_wt = sum(nonzero.values())
    if total_wt <= 0:
        return dict(d_w=0.0, EN_FeP=0.0)

    # Weighted d-electron count
    d_w = sum(
        wt * D_ELECTRONS.get(COL_TO_ELEM.get(col, ""), 0)
        for col, wt in nonzero.items()
    ) / total_wt

    # EN_FeP: electronegativity excluding the alkali promoter (K is always excluded)
    non_alk       = {col: wt for col, wt in wt_dict.items() if col != "K_wt"}
    total_non_alk = sum(non_alk.values())
    EN_FeP = (
        sum(wt * EN.get(COL_TO_ELEM.get(col, ""), 0)
            for col, wt in non_alk.items()) / total_non_alk
        if total_non_alk > 0 else 0.0
    )

    return dict(d_w=d_w, EN_FeP=EN_FeP)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_ME_LIST = list(ME_POOL.keys())


def load_models(paths: dict) -> dict:
    global FEATURE_COLUMNS, N_FEATURES, _COL_IDX
    models = {}
    for name, path in paths.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"  [✓] Loaded  →  {name}  ({path})")
    first_model     = next(iter(models.values()))
    FEATURE_COLUMNS = list(first_model.get_booster().feature_names)
    N_FEATURES      = len(FEATURE_COLUMNS)
    _COL_IDX        = {col: i for i, col in enumerate(FEATURE_COLUMNS)}
    print(f"  [✓] FEATURE_COLUMNS locked ({N_FEATURES} features)")
    return models


def build_feature_matrix(X_cont: np.ndarray) -> np.ndarray:
    """
    Decision variables — 10 total:
        x[0]  Fe_wt          FE_WT_RANGE         active metal (pre-norm)
        x[1]  K_wt           K_WT_RANGE           alkali promoter (pre-norm)
        x[2]  Al_wt          AL_WT_RANGE          structural promoter (pre-norm)
        x[3]  Me1 index      [0, n_me-1]          rounded integer
        x[4]  Me1_wt         ME_WT_RANGE          additional promoter 1 (pre-norm)
        x[5]  Me2 index      [0, n_me-1]          rounded integer
        x[6]  Me2_wt         ME_WT_RANGE          additional promoter 2 (pre-norm)
        x[7]  temp_C         TEMP_C_RANGE         optimised
        x[8]  P_MPa          P_MPA_RANGE          optimised
        x[9]  GHSV           GHSV_RANGE           optimised
    """
    n_me = len(_ME_LIST)
    rows = []

    for x in X_cont:
        # ── Composition (pre-normalisation) ──────────────────────────────────
        fe_wt   = float(x[0])
        k_wt    = float(x[1])
        al_wt   = float(x[2])
        me1_idx = int(np.round(np.clip(x[3], 0, n_me - 1)))
        me1_wt  = float(x[4])
        me2_idx = int(np.round(np.clip(x[5], 0, n_me - 1)))
        me2_wt  = float(x[6])

        # Ensure Me1 ≠ Me2
        if me2_idx == me1_idx:
            me2_idx = (me2_idx + 1) % n_me

        # ── Operating conditions ─────────────────────────────────────────────
        temp_c = float(x[7])
        p_mpa  = float(x[8])
        ghsv   = float(x[9])

        me1_col = ME_POOL[_ME_LIST[me1_idx]]
        me2_col = ME_POOL[_ME_LIST[me2_idx]]

        # Assemble weight dict (zero all, then fill)
        wt_dict = {col: 0.0 for col in COL_TO_ELEM}
        wt_dict["Fe_wt"]  = fe_wt
        wt_dict["K_wt"]   = k_wt
        wt_dict["Al_wt"]  = al_wt
        wt_dict[me1_col]  = me1_wt
        wt_dict[me2_col]  = me2_wt

        # Normalise to 100 wt%
        total   = sum(wt_dict.values())
        wt_dict = {col: val / total * 100.0 for col, val in wt_dict.items()}

        # Descriptors
        desc = _compute_descriptors(wt_dict)

        fv = np.zeros(N_FEATURES)

        # Fixed synthesis conditions
        fv[_COL_IDX["support"]]      = SUPPORT_CODE
        fv[_COL_IDX["synth"]]        = SYNTH_CODE
        fv[_COL_IDX["calc_temp_C"]]  = CALC_TEMP_C
        fv[_COL_IDX["reduc_temp_C"]] = REDUC_TEMP_C
        fv[_COL_IDX["H2_CO2"]]       = H2_CO2

        # Optimised operating conditions
        fv[_COL_IDX["temp_C"]] = temp_c
        fv[_COL_IDX["P_MPa"]]  = p_mpa
        fv[_COL_IDX["GHSV"]]   = ghsv

        # Descriptors
        fv[_COL_IDX["d_w"]]    = desc["d_w"]
        fv[_COL_IDX["EN_FeP"]] = desc["EN_FeP"]

        # Composition weights
        for col, val in wt_dict.items():
            if col in _COL_IDX:
                fv[_COL_IDX[col]] = val

        rows.append(fv)

    return np.array(rows, dtype=float)


def decode_solution(x: np.ndarray) -> dict:
    n_me    = len(_ME_LIST)
    me1_idx = int(np.round(np.clip(x[3], 0, n_me - 1)))
    me2_idx = int(np.round(np.clip(x[5], 0, n_me - 1)))
    if me2_idx == me1_idx:
        me2_idx = (me2_idx + 1) % n_me

    me1_col = ME_POOL[_ME_LIST[me1_idx]]
    me2_col = ME_POOL[_ME_LIST[me2_idx]]

    raw = {
        "Fe_wt": float(x[0]),
        "K_wt":  float(x[1]),
        "Al_wt": float(x[2]),
        me1_col: float(x[4]),
        me2_col: float(x[6]),
    }
    total = sum(raw.values())
    norm  = {col: val / total * 100.0 for col, val in raw.items()}

    temp_c = round(float(x[7]), 1)
    p_mpa  = round(float(x[8]), 3)
    ghsv   = round(float(x[9]), 0)

    return {
        "Fe_wt%"          : round(norm.get("Fe_wt", 0.0), 2),
        "K_wt%"           : round(norm.get("K_wt",  0.0), 2),
        "Al_wt%"          : round(norm.get("Al_wt", 0.0), 2),
        "Me1_element"     : _ME_LIST[me1_idx],
        "Me1_wt%"         : round(norm.get(me1_col, 0.0), 2),
        "Me2_element"     : _ME_LIST[me2_idx],
        "Me2_wt%"         : round(norm.get(me2_col, 0.0), 2),
        "Reaction_temp_C" : temp_c,
        "Pressure_MPa"    : p_mpa,
        "GHSV_mL_g_h"     : ghsv,
        "Calc_temp_C"     : CALC_TEMP_C,
        "Reduc_temp_C"    : REDUC_TEMP_C,
        "H2_CO2_ratio"    : H2_CO2,
        "Catalyst_type"   : "Bulk",
        "Synthesis"       : "Co-precipitation",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PARETO TRACKER  (unchanged from v9)
# ══════════════════════════════════════════════════════════════════════════════

class ParetoTracker:
    def __init__(self, ref_point: np.ndarray):
        self._evals: list     = []
        self._hv_indicator    = HV(ref_point=ref_point)
        self.hv_history: list = []
        self.eval_nums:  list = []

    @staticmethod
    def _fast_nondominated(F_neg: np.ndarray) -> np.ndarray:
        n  = len(F_neg)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            dominated = (np.all(F_neg[i] <= F_neg, axis=1) &
                         np.any(F_neg[i] <  F_neg, axis=1))
            dominated[i] = False
            nd[dominated] = False
        return nd

    def add(self, x: np.ndarray, F: np.ndarray):
        self._evals.append((np.asarray(x, dtype=float),
                             np.asarray(F, dtype=float)))
        all_F_neg = -np.vstack([e[1] for e in self._evals])
        nd_mask   = self._fast_nondominated(all_F_neg)
        hv_val    = (self._hv_indicator(all_F_neg[nd_mask])
                     if nd_mask.any() else 0.0)
        self.hv_history.append(float(hv_val))
        self.eval_nums.append(len(self._evals))

    @property
    def all_X(self): return np.vstack([e[0] for e in self._evals])
    @property
    def all_F(self): return np.vstack([e[1] for e in self._evals])
    @property
    def pareto_mask(self): return self._fast_nondominated(-self.all_F)
    @property
    def pareto_X(self): return self.all_X[self.pareto_mask]
    @property
    def pareto_F(self): return self.all_F[self.pareto_mask]
    @property
    def pareto_F_neg(self): return -self.pareto_F


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PARETO OBJECTIVE & CHEBYSHEV SCALARISATION  (unchanged from v9)
# ══════════════════════════════════════════════════════════════════════════════

def make_scalarised_objective(models, weights, tracker, rho=RHO):
    def objective(x: list) -> float:
        x_arr  = np.array(x, dtype=float).reshape(1, -1)
        X_feat = build_feature_matrix(x_arr)
        co2 = float(np.clip(models["CO2_conversion"].predict(X_feat), 0, 100)[0])
        c5y = float(np.clip(models["C5+_yield"].predict(X_feat),      0, 100)[0])
        F      = np.array([co2, c5y])
        loss   = 100.0 - F
        tracker.add(x_arr[0], F)
        return float(np.max(weights * loss) + rho * np.dot(weights, loss))
    return objective


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SINGLE SEED — N_RESTARTS ParEGO BO restarts
# ══════════════════════════════════════════════════════════════════════════════

def run_single_bo(models, seed,
                  n_restarts=N_RESTARTS, n_calls=N_CALLS_PER_RESTART,
                  n_initial=N_INITIAL_PER_RESTART, verbose=False):
    np.random.seed(seed)
    n_me = len(_ME_LIST)

    # 10 decision variables: Fe / K / Al / Me1-idx / Me1-wt / Me2-idx / Me2-wt
    #                       / temp_C / P_MPa / GHSV
    space = [
        Real(FE_WT_RANGE[0],  FE_WT_RANGE[1],  name="x0"),  # Fe_wt
        Real(K_WT_RANGE[0],   K_WT_RANGE[1],   name="x1"),  # K_wt  (always K)
        Real(AL_WT_RANGE[0],  AL_WT_RANGE[1],  name="x2"),  # Al_wt (always Al)
        Real(0.0,             float(n_me - 1), name="x3"),  # Me1 index
        Real(ME_WT_RANGE[0],  ME_WT_RANGE[1],  name="x4"),  # Me1 wt
        Real(0.0,             float(n_me - 1), name="x5"),  # Me2 index
        Real(ME_WT_RANGE[0],  ME_WT_RANGE[1],  name="x6"),  # Me2 wt
        Real(TEMP_C_RANGE[0], TEMP_C_RANGE[1], name="x7"),  # temp_C
        Real(P_MPA_RANGE[0],  P_MPA_RANGE[1],  name="x8"),  # P_MPa
        Real(GHSV_RANGE[0],   GHSV_RANGE[1],   name="x9"),  # GHSV
    ]

    tracker     = ParetoTracker(ref_point=HV_REF_POINT)
    weight_grid = np.array([
        (i / (n_restarts - 1), 1.0 - i / (n_restarts - 1))
        for i in range(n_restarts)
    ])

    for r in range(n_restarts):
        w   = weight_grid[r]
        obj = make_scalarised_objective(models, w, tracker)
        gp_minimize(
            obj,
            dimensions       = space,
            n_calls          = n_calls,
            n_initial_points = n_initial,
            acq_func         = "EI",
            acq_optimizer    = "lbfgs",
            random_state     = seed + r * 7,
            verbose          = False,
            noise            = 1e-6,
        )
        if verbose and (r % 5 == 0 or r == n_restarts - 1):
            hv_now = tracker.hv_history[-1] if tracker.hv_history else 0.0
            n_par  = int(tracker.pareto_mask.sum())
            print(f"    Restart {r+1:>3}/{n_restarts}  |  "
                  f"Evals: {len(tracker.hv_history):>5}  |  "
                  f"Pareto: {n_par:>4}  |  HV: {hv_now:.4f}", flush=True)

    return tracker.pareto_F_neg, tracker


# ══════════════════════════════════════════════════════════════════════════════
# 8.  QUALITY METRICS  (unchanged from v9)
# ══════════════════════════════════════════════════════════════════════════════

def compute_gdplus_igdplus(all_fronts):
    combined = np.vstack(all_fronts)
    def fast_nd(F):
        n  = len(F); nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]: continue
            dom = (np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1))
            dom[i] = False; nd[dom] = False
        return nd
    ref_front    = combined[fast_nd(combined)]
    hv_indicator = HV(ref_point=HV_REF_POINT)
    gd_s, igd_s, hv_s = [], [], []
    for F in all_fronts:
        gd_s.append(GDPlus(ref_front)(F))
        igd_s.append(IGDPlus(ref_front)(F))
        hv_s.append(hv_indicator(F))
    return {"reference_front": ref_front, "gd_plus": gd_s,
            "igd_plus": igd_s, "hv_per_seed": hv_s}


# ══════════════════════════════════════════════════════════════════════════════
# 9.  VISUALISATION  (updated labels for Fe/K/Al/Me1/Me2 architecture)
# ══════════════════════════════════════════════════════════════════════════════

def plot_hv_convergence(tracker, run_ts, out_dir):
    save_path = os.path.join(out_dir, f"hv_convergence_{run_ts}.png")
    fig, ax   = plt.subplots(figsize=(11, 5))
    ax.plot(tracker.eval_nums, tracker.hv_history,
            color="steelblue", linewidth=1.8, alpha=0.9, label="Cumulative HV")
    ax.fill_between(tracker.eval_nums, 0, tracker.hv_history,
                    alpha=0.12, color="steelblue")
    for r in range(1, N_RESTARTS):
        bnd = r * N_CALLS_PER_RESTART
        if bnd <= tracker.eval_nums[-1]:
            ax.axvline(x=bnd, color="gray", linestyle="--",
                       linewidth=0.7, alpha=0.5,
                       label="Restart boundary" if r == 1 else None)
    final_hv = tracker.hv_history[-1]
    ax.axhline(y=final_hv, color="tomato", linestyle=":",
               linewidth=1.2, label=f"Final HV = {final_hv:.4f}")
    ax.set_xlabel("Number of Function Evaluations", fontsize=11)
    ax.set_ylabel("Hypervolume (HV)", fontsize=11)
    ax.set_title(
        "BO (ParEGO) — HV Convergence  |  Fe/K/Al + Me1/Me2 Catalyst Architecture\n"
        f"{N_RESTARTS} restarts × {N_CALLS_PER_RESTART} evals/restart  |  10 decision variables",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [✓] HV convergence  →  {save_path}")
    plt.show()


def plot_2obj_pareto_with_front(F_pareto, ref_front, df_pareto, save_path, top_n=10):
    co2, yld = F_pareto[:, 0], F_pareto[:, 1]
    ref_co2  = -ref_front[:, 0];  ref_yld = -ref_front[:, 1]
    top_orig_idx = df_pareto.head(top_n)["_orig_idx"].tolist()
    top_mask     = np.zeros(len(F_pareto), dtype=bool)
    for orig in top_orig_idx: top_mask[orig] = True
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(ref_co2, ref_yld, c="darkorange", s=18, alpha=0.35,
               marker="^", label="Approx. true front (multi-seed union)")
    norm = plt.Normalize(yld.min(), yld.max())
    sc   = ax.scatter(co2[~top_mask], yld[~top_mask], c=yld[~top_mask],
                      cmap=cm.plasma, norm=norm, s=35, alpha=0.55,
                      edgecolors="none", label="BO Pareto solutions")
    ax.scatter(co2[top_mask], yld[top_mask], c="lime", s=130, alpha=1.0,
               edgecolors="black", linewidths=0.8, zorder=5, label=f"Top-{top_n}")
    for rank, orig in enumerate(top_orig_idx, 1):
        ax.annotate(f"#{rank}", (co2[orig], yld[orig]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="C5+ Yield (%)")
    ax.set_xlabel("CO₂ Conversion (%)", fontsize=11)
    ax.set_ylabel("C5+ Yield (%)", fontsize=11)
    ax.set_title(
        "Pareto Front — BO-ParEGO  |  Fe–K–Al–Me1–Me2 System\n"
        f"Top-{top_n} highlighted  |  Orange = multi-seed approx. true front",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [✓] 2-obj Pareto  →  {save_path}")
    plt.show()


def plot_composition_heatmap(df_top, save_path):
    numeric_cols = [
        "Fe_wt%", "K_wt%", "Al_wt%", "Me1_wt%", "Me2_wt%",
        "Reaction_temp_C", "Pressure_MPa", "GHSV_mL_g_h",
        "Calc_temp_C", "Reduc_temp_C", "H2_CO2_ratio",
    ]
    df_heat       = df_top[numeric_cols].copy()
    df_heat.index = [f"Cat-{i+1}" for i in range(len(df_heat))]
    df_norm       = (df_heat - df_heat.min()) / (df_heat.max() - df_heat.min() + 1e-9)
    fig, ax       = plt.subplots(figsize=(15, max(6, len(df_top) * 0.4)))
    sns.heatmap(df_norm, annot=df_heat.round(1), fmt="g",
                cmap="YlOrRd", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Normalised value"})
    ax.set_title(
        f"Top-{len(df_top)} Pareto Optimal Catalysts — Fe/K/Al/Me1/Me2 System",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [✓] Composition heatmap  →  {save_path}")
    plt.show()


def plot_me_frequency(df_pareto, save_path):
    """Bar chart: how often each Me1/Me2 element appears in the Pareto front."""
    me_counts = (
        pd.concat([df_pareto["Me1_element"], df_pareto["Me2_element"]])
        .value_counts()
    )
    colors = ["#2196F3" if e in ["Cu", "Na"] else
              "#FF9800" if e in ["Zn", "Mn"] else "#F44336"
              for e in me_counts.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    me_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Promoter element (Me1 or Me2)", fontsize=11)
    ax.set_ylabel("Frequency in Pareto front", fontsize=11)
    ax.set_title("Promoter Selection Frequency — Pareto Front\n"
                 "(blue=positive, orange=neutral, red=negative in dataset)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [✓] Me frequency plot  →  {save_path}")
    plt.show()


def plot_operating_conditions(df_top, save_path):
    score = df_top["Composite_Score"].values
    norm  = plt.Normalize(score.min(), score.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [
        ("Reaction_temp_C", "Pressure_MPa"),
        ("Reaction_temp_C", "GHSV_mL_g_h"),
        ("Pressure_MPa",    "GHSV_mL_g_h"),
    ]
    for ax, (xcol, ycol) in zip(axes, pairs):
        sc = ax.scatter(df_top[xcol], df_top[ycol], c=score, cmap="plasma",
                        norm=norm, s=40, alpha=0.8, edgecolors="none")
        ax.set_xlabel(xcol, fontsize=10); ax.set_ylabel(ycol, fontsize=10)
        ax.grid(True, alpha=0.25)
    plt.colorbar(sc, ax=axes[-1], label="Composite Score")
    fig.suptitle("Optimised Operating Conditions — Fe/K/Al/Me1/Me2 Pareto Front",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  [✓] Operating conditions  →  {save_path}")
    plt.show()


def plot_eval_distribution(tracker, run_ts, out_dir):
    all_F   = tracker.all_F
    co2_all = all_F[:, 0]; yld_all = all_F[:, 1]
    par_F   = tracker.pareto_F
    co2_par = par_F[:, 0]; yld_par = par_F[:, 1]
    is_par  = tracker.pareto_mask

    csv_path = os.path.join(out_dir, f"eval_distribution_{run_ts}.csv")
    pd.DataFrame({
        "Eval_index":        tracker.eval_nums,
        "CO2_Conversion_%":  co2_all,
        "C5plus_Yield_%":    yld_all,
        "Is_Pareto":         is_par.astype(int),
    }).to_csv(csv_path, index=False)

    save_path = os.path.join(out_dir, f"eval_distribution_{run_ts}.png")
    colors    = np.linspace(0.2, 1.0, len(co2_all))
    fig, ax   = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(co2_all, yld_all, c=colors, cmap=plt.cm.Blues,
                    s=12, alpha=0.5, linewidths=0, label="All BO evaluations")
    ax.scatter(co2_par, yld_par, c="tomato", s=60, alpha=0.9,
               edgecolors="black", linewidths=0.5, zorder=5, label="Pareto solutions")
    plt.colorbar(sc, ax=ax, label="Evaluation order (darker = earlier)")
    ax.set_xlabel("CO₂ Conversion (%)", fontsize=11)
    ax.set_ylabel("C5+ Yield (%)", fontsize=11)
    ax.set_title("BO Evaluation Distribution — Fe/K/Al/Me1/Me2 System",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 10.  REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def write_metrics_report(metrics, seeds, main_hv_history, run_ts, out_dir):
    path  = os.path.join(out_dir, f"metrics_report_{run_ts}.txt")
    gd, igd, hv = metrics["gd_plus"], metrics["igd_plus"], metrics["hv_per_seed"]
    hv_arr  = np.array(main_hv_history); n_evals = len(hv_arr)
    p90_idx = max(0, int(n_evals * 0.90) - 1)
    p80_idx = max(0, int(n_evals * 0.80) - 1)
    hv_chg_pct   = abs(hv_arr[p90_idx] - hv_arr[p80_idx]) / (hv_arr[p80_idx] + 1e-9) * 100
    hv_cv        = np.std(hv) / (np.mean(hv) + 1e-9) * 100
    convergence_status = ("CONVERGED" if hv_chg_pct < 1.0
                          else f"NOT CONVERGED — HV changing {hv_chg_pct:.2f}%")
    assessment = (
        "EXCELLENT" if hv_cv <  2.0 and hv_chg_pct <  1.0 else
        "GOOD"      if hv_cv <  5.0 and hv_chg_pct <  3.0 else
        "ACCEPTABLE — consider more restarts" if hv_cv < 10.0
        else "POOR — increase N_RESTARTS or N_CALLS"
    )
    sep  = "=" * 70
    sep2 = "-" * 70
    lines = [
        sep, "  BO (ParEGO) — QUALITY METRICS", sep,
        f"  Architecture  : Fe (active) / K (alkali) / Al (structural) / Me1 / Me2",
        f"  Decision vars : 10  (Fe_wt, K_wt, Al_wt, Me1/Me2 idx+wt, T, P, GHSV)",
        f"  FE range      : {FE_WT_RANGE}  wt%",
        f"  K range       : {K_WT_RANGE}   wt%",
        f"  Al range      : {AL_WT_RANGE}  wt%",
        f"  Me pool       : {list(ME_POOL.keys())}",
        f"  Me wt range   : {ME_WT_RANGE}  wt%",
        f"  Temp range    : {TEMP_C_RANGE} °C",
        f"  P range       : {P_MPA_RANGE}  MPa",
        f"  GHSV range    : {GHSV_RANGE}   mL/g/h",
        f"  CALC_TEMP_C   : {CALC_TEMP_C} °C (fixed)",
        f"  REDUC_TEMP_C  : {REDUC_TEMP_C} °C (fixed)",
        "",
        sep2,
        f"  {'Seed':<8} {'HV':>10} {'GD+':>12} {'IGD+':>12}",
        *[f"  {s:<8} {hv_v:>10.4f} {gd_v:>12.6f} {igd_v:>12.6f}"
          f"{'  ◀ main' if s == SEED else ''}"
          for s, hv_v, gd_v, igd_v in zip(seeds, hv, gd, igd)],
        "",
        f"  Mean HV  : {np.mean(hv):.6f}  ±  {np.std(hv):.6f}",
        f"  CV(%)    : {hv_cv:.2f}%",
        f"  Convergence : {convergence_status}",
        f"  Assessment  : {assessment}",
        sep, "  END", sep,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [✓] Metrics report  →  {path}")
    return path


def write_report(df_pareto, F_pareto, run_ts, out_dir):
    path = os.path.join(out_dir, f"final_report_{run_ts}.txt")
    top10 = df_pareto.head(10)
    sep   = "=" * 80
    sep2  = "-" * 80
    display_cols = [
        "Fe_wt%", "K_wt%", "Al_wt%",
        "Me1_element", "Me1_wt%", "Me2_element", "Me2_wt%",
        "Reaction_temp_C", "Pressure_MPa", "GHSV_mL_g_h",
        "Predicted_CO2_Conversion_%", "Predicted_C5+_Yield_%", "Composite_Score",
    ]
    lines = [
        sep, "  BO (ParEGO) — FINAL REPORT  |  Fe/K/Al/Me1/Me2 System", sep,
        f"  Run: {run_ts}   Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "", "  TOP-10 CATALYSTS", sep2,
    ]
    for rank, row in top10.iterrows():
        lines += [
            f"  #{rank}  Fe:{row['Fe_wt%']}%  K:{row['K_wt%']}%  "
            f"Al:{row['Al_wt%']}%  "
            f"{row['Me1_element']}:{row['Me1_wt%']}%  "
            f"{row['Me2_element']}:{row['Me2_wt%']}%",
            f"     T={row['Reaction_temp_C']}°C  P={row['Pressure_MPa']}MPa  "
            f"GHSV={int(row['GHSV_mL_g_h'])}",
            f"     CO2={row['Predicted_CO2_Conversion_%']:.2f}%  "
            f"Yield={row['Predicted_C5+_Yield_%']:.2f}%  "
            f"Score={row['Composite_Score']:.4f}", "",
        ]
    lines += [sep2, f"  FULL TOP-{TOP_N}", sep2,
              df_pareto[display_cols].head(TOP_N).to_string(),
              sep, "  END", sep]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [✓] Final report  →  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_optimization():
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    def out(fn): return os.path.join(OUTPUT_DIR, fn)

    print("=" * 70)
    print("  BO (ParEGO) — Fe/K/Al + Me1/Me2 Catalyst Architecture")
    print("  Active metal  : Fe")
    print("  Alkali        : K  (fixed, range 1–25 wt%)")
    print("  Structural    : Al (fixed, range 3–25 wt%)")
    print(f"  Me pool       : {list(ME_POOL.keys())}")
    print(f"  Temp  {TEMP_C_RANGE[0]}–{TEMP_C_RANGE[1]} °C  |  "
          f"P {P_MPA_RANGE[0]}–{P_MPA_RANGE[1]} MPa  |  "
          f"GHSV {int(GHSV_RANGE[0])}–{int(GHSV_RANGE[1])} mL/g/h")
    print(f"  {N_RESTARTS} restarts × {N_CALLS_PER_RESTART} calls "
          f"= {N_RESTARTS * N_CALLS_PER_RESTART} total evals")
    print("=" * 70)

    print("\n[1/5]  Loading surrogate models …")
    models = load_models(MODEL_PATHS)

    print(f"\n[2/5]  Running BO across {len(STABILITY_SEEDS)} seeds …")
    all_fronts, all_trackers = [], []
    for seed in STABILITY_SEEDS:
        tag = "MAIN" if seed == SEED else "    "
        print(f"  [{tag}]  Seed {seed} …", flush=True)
        F_neg, tracker = run_single_bo(models, seed, verbose=(seed == SEED))
        all_fronts.append(F_neg)
        all_trackers.append(tracker)
        final_hv = tracker.hv_history[-1] if tracker.hv_history else 0.0
        print(f"         Pareto: {len(F_neg)}   HV: {final_hv:.6f}")

    main_idx     = STABILITY_SEEDS.index(SEED)
    main_tracker = all_trackers[main_idx]

    print("\n[3/5]  Computing HV, GD+, IGD+ …")
    metrics = compute_gdplus_igdplus(all_fronts)
    print(f"\n  {'Seed':<8} {'HV':>10} {'GD+':>12} {'IGD+':>12}")
    print(f"  {'-' * 46}")
    for s, hv_v, gd_v, igd_v in zip(
        STABILITY_SEEDS, metrics["hv_per_seed"],
        metrics["gd_plus"], metrics["igd_plus"]
    ):
        print(f"  {s:<8} {hv_v:>10.4f} {gd_v:>12.6f} {igd_v:>12.6f}"
              f"{'  ◀ main' if s == SEED else ''}")

    print("\n[4/5]  Building result table …")
    X_pareto = main_tracker.pareto_X
    F_pareto = main_tracker.pareto_F
    records  = []
    for i in range(len(X_pareto)):
        cat = decode_solution(X_pareto[i])
        cat["Predicted_CO2_Conversion_%"] = round(float(F_pareto[i, 0]), 2)
        cat["Predicted_C5+_Yield_%"]      = round(float(F_pareto[i, 1]), 2)
        records.append(cat)

    df_pareto = pd.DataFrame(records)
    df_pareto["Composite_Score"] = (
        df_pareto["Predicted_CO2_Conversion_%"]
        + df_pareto["Predicted_C5+_Yield_%"]
    ) / 2.0
    df_pareto["_orig_idx"] = range(len(df_pareto))
    df_pareto.sort_values("Composite_Score", ascending=False, inplace=True)
    df_pareto.reset_index(drop=True, inplace=True)
    df_pareto.index += 1

    df_pareto.to_csv(out("pareto_optimal_catalysts.csv"), index_label="Rank")

    print("[5/5]  Plots and reports …")
    plot_hv_convergence(main_tracker, run_ts, OUTPUT_DIR)
    plot_2obj_pareto_with_front(F_pareto, metrics["reference_front"],
                                df_pareto, out(f"pareto_2obj_{run_ts}.png"), top_n=10)
    plot_composition_heatmap(df_pareto.head(TOP_N),
                             out(f"pareto_heatmap_{run_ts}.png"))
    plot_me_frequency(df_pareto, out(f"me_frequency_{run_ts}.png"))  # NEW
    plot_operating_conditions(df_pareto.head(TOP_N),
                              out(f"operating_conditions_{run_ts}.png"))
    plot_eval_distribution(main_tracker, run_ts, OUTPUT_DIR)

    write_report(df_pareto, F_pareto, run_ts, OUTPUT_DIR)
    write_metrics_report(metrics, STABILITY_SEEDS,
                         main_tracker.hv_history, run_ts, OUTPUT_DIR)

    print(f"\n{'=' * 70}")
    print(f"  Done.  Outputs: {OUTPUT_DIR}")
    print(f"  Pareto solutions: {len(df_pareto)}")
    print(f"  Total BO evals  : {len(main_tracker.hv_history)}")
    print(f"{'=' * 70}\n")
    return df_pareto, main_tracker, metrics


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df_results, bo_tracker, quality_metrics = run_optimization()
