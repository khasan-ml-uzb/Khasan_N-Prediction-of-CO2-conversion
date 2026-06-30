# Khasan_N-Prediction-of-CO2-conversion
Herein I built a model to predict optimal catalyst to produce gasoline product by CO2 hydrogenation reaction

This work is related to the publication in **Fuel 427 (2027) 139918**

## Title: Mitigating the trade-off between selectivity and stability in CO2 hydrogenation using ultra-high potassium loading on iron catalysts
Authors: Khasan Nasriddinov, Mansurbek Urol ugli Abdullaev, Hae-Gu Park, Hyung-Ki Min, Ki-Won Jun, Ji-Eun Min, Jeong-Rang Kim, Chundong Zhang, , Seok Ki Kim

DOI: https://doi.org/10.1016/j.fuel.2026.139918
<img width="1244" height="886" alt="image" src="https://github.com/user-attachments/assets/d00bf073-bd5b-4831-8f1f-31278f62f2ff" />


### Highlights
> Ultra-high K loading (30 wt%) on FeCu/γ-Al2O3 breaks the CO2 hydrogenation selectivity-stability trade-off: 52.6% C5+, 100-h stability.
> Potassium-driven electronic modification speeds Fe5C2 carburization and suppresses hydrogenation, shifting output to long-chain hydrocarbons.
> A K-stabilized mixed-phase Fe3O4/Fe5C2 structure resists sintering, oxidation, and carbon-induced deactivation, preserving activity.
> γ-Al2O3 disperses excess potassium and preserves catalyst porosity even at ultra-high loadings, preventing active-site blockage.
> Long-term runs show 15.6% C5+ yield with methane cut to 12%, setting a new design paradigm for industrial CO2-to-fuel catalysts.

## Interactive Streamlit app (`streamlita_app.py`)

An interactive web app that deploys the trained XGBoost surrogate models and the
Bayesian-optimization catalyst search. Change **Temperature**, **Pressure** and
**GHSV**, and the app recommends the **optimal catalyst** (Fe–K–Al + two
selectable promoters) for those conditions, predicting CO₂ conversion, C5+
selectivity and C5+ yield.

**Run locally**
```bash
pip install -r requirements.txt
python XGBoost.py            # trains models → *_model.pkl
streamlit run streamlita_app.py
```

**Deploy on Streamlit Cloud**
1. Commit `streamlita_app.py`, `requirements.txt`, and the trained
   `CO2_conv_model.pkl`, `C5plus_sel_model.pkl`, `C5plus_yield_model.pkl`
   files (the app searches the repo root, `models/`, and
   `outputs/xgboost_v3/models/`).
2. On [share.streamlit.io](https://share.streamlit.io), point the app file to
   `streamlita_app.py`.

**Features**
- 🎯 *Optimal catalyst* — recommends the best composition for the chosen
  conditions via a fast vectorised random search over the surrogate models.
- 🧪 *Manual prediction* — predict performance for a specific catalyst.
- 📈 *Condition sweep* — see how the optimum shifts as one of T / P / GHSV varies.
