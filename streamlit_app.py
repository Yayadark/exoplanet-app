# ----------------------------- 
# Streamlit app: KOI+K2 model
# - Users can enter ANY subset of features
# - Toggle: "Fill missing with medians" (warns and lists filled fields)
# - Outputs probability + label (tau=0.60)
# -----------------------------
import json, joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============ 1) Load model + artifacts ============
# These are what you saved earlier from Colab:
#   model_koi_k2.joblib  and  artifacts_koi_k2.json
@st.cache_resource
def load_assets():
    model = joblib.load("model_koi_k2.joblib")
    with open("artifacts_koi_k2.json", "r") as f:
        artifacts = json.load(f)
    return model, artifacts

model, artifacts = load_assets()
FEATURES = artifacts["feature_names"]      # exact feature order the model expects
MEDIANS  = artifacts["medians"]            # {feature: median_value}
TAU      = artifacts.get("tau", 0.60)      # operating threshold

# -------- Pretty labels to match KOI/K2/TESS tables --------
LABEL_MAP = {
    "period_days": "Orbital Period [days]",
    "duration_hours": "Transit Duration [hours]",
    "depth_ppm": "Transit Depth [ppm]",
    "snr": "Transit S/N",
    "impact_parameter": "Impact Parameter (b)",
    "a_over_rstar": "a/R★ (Semi-major axis / Stellar radius)",
    "radius_re": "Planet Radius [R⊕]",
    "teq_k": "Planet Equilibrium Temperature [K]",
    "inclination_deg": "Inclination [deg]",
    "num_transits": "Number of Transits",
    "planet_number": "TCE / Planet Number",
    "teff_k": "Stellar Effective Temperature [K]",
    "logg_cgs": "Stellar Surface Gravity log g [cgs]",
    "feh_dex": "Stellar Metallicity [Fe/H] (dex)",
    "rstar_rsun": "Stellar Radius [R☉]",
    "mstar_msun": "Stellar Mass [M☉]",
    "rho_star": "Fitted Stellar Density [g cm⁻³]",
    "mag": "Magnitude (Kepler/TESS band)",
}

# Optional one-line tooltips
HELP_MAP = {
    "depth_ppm": "If your table shows depth in %, convert: ppm = % × 10,000.",
    "impact_parameter": "0 = center crossing; ~1 = grazing transit.",
    "a_over_rstar": "Checks physical consistency of period & duration with the star.",
    "snr": "Signal-to-noise of the detected transit.",
    "mag": "Use Kepler mag for KOI/K2 or TESS mag for TESS; blank = median fill.",
}

# (Optional) friendlier display order – only keeps features your model actually has
FEATURES_DISPLAY_ORDER = [
    "period_days","duration_hours","depth_ppm",
    "snr","impact_parameter","a_over_rstar",
    "radius_re","teq_k","inclination_deg",
    "teff_k","mag","num_transits","planet_number",
    "rho_star","logg_cgs","feh_dex","rstar_rsun","mstar_msun",
]
FEATURES_FOR_UI = [f for f in FEATURES_DISPLAY_ORDER if f in FEATURES] + \
                  [f for f in FEATURES if f not in FEATURES_DISPLAY_ORDER]

# ============ 2) Small helper to predict ============
def predict_one(input_dict, model, feature_order, medians, tau):
    """
    - input_dict: user-provided numeric values (strings already parsed to floats)
    - fills any missing keys using medians
    - returns: probability, label, and which fields were auto-filled
    """
    row = {}
    filled = []  # keep track of which features we auto-filled

    for col in feature_order:
        if col in input_dict and input_dict[col] is not None:
            row[col] = float(input_dict[col])
        else:
            row[col] = float(medians.get(col, 0.0))
            filled.append(col)

    X_one = pd.DataFrame([row], columns=feature_order)

    try:
        p = float(model.predict_proba(X_one)[:, 1][0])
    except AttributeError:
        p = float(model.predict(X_one)[0])

    label = "High-confidence candidate" if p >= tau else "Candidate / Likely FP"
    return p, label, filled, row

# ============ 3) UI layout ============
st.set_page_config(page_title="Exoplanet Classifier (KOI+K2)", layout="centered")
st.title("Exoplanet Classifier — KOI+K2 Model")
st.caption("Enter what you know. Optionally auto-fill any missing fields with training-set medians.")

with st.sidebar:
    st.subheader("Settings")
    use_medians = st.checkbox(
        "Fill any missing fields with medians (recommended)",
        value=True,
        help="If ON: we auto-fill blanks using medians from training data and warn you which ones."
    )
    st.write(f"Threshold τ = **{TAU:.2f}** (Conservative mode)")

st.subheader("Input features")
st.write("Leave fields blank if unknown. You can fill the high-impact ones at least (snr, duration_hours, impact_parameter, radius_re, depth_ppm, mag).")

# We’ll use text inputs so blanks are allowed, then parse to float
def parse_float(x):
    if x is None or str(x).strip() == "":
        return None
    try:
        return float(x)
    except:
        return None  # treat invalid text as missing

cols = st.columns(2)
inputs = {}
inputs = {}

def parse_float(x):
    if x is None or str(x).strip() == "":
        return None
    try:
        return float(x)
    except:
        return None

for i, feat in enumerate(FEATURES_FOR_UI if 'FEATURES_FOR_UI' in globals() else FEATURES):
    label = LABEL_MAP.get(feat, feat)                 # pretty label
    help_txt = HELP_MAP.get(feat, None)               # optional tooltip
    placeholder = f"(median ≈ {MEDIANS.get(feat, 0):.3g})"
    with cols[i % 2]:
        val = st.text_input(label, value="", placeholder=placeholder, help=help_txt)
        inputs[feat] = parse_float(val)

# Figure out which are missing
missing = [k for k, v in inputs.items() if v is None]

# ============ 4) Predict button ============
if st.button("Predict"):
    if (not use_medians) and len(missing) > 0:
        # User chose NOT to auto-fill → require all fields
        st.error(
            f"You turned OFF median auto-fill. Please provide values for ALL features. "
            f"Missing ({len(missing)}): {', '.join(missing[:12])}{'...' if len(missing)>12 else ''}"
        )
        st.stop()

    # If auto-fill is ON, we’ll fill the missing ones with medians and warn
    p, label, filled, used_row = predict_one(
        inputs, model, FEATURES, MEDIANS, TAU
    )

    # Show warning only if we actually filled some
    if use_medians and len(filled) > 0:
        st.warning(
            f"Filled {len(filled)} missing field(s) with training medians: "
            f"{', '.join(filled[:12])}{'...' if len(filled)>12 else ''}"
        )

    # ============ 5) Results ============
    st.markdown("---")
    st.subheader("Result")
    st.metric(label="Probability p(planet)", value=f"{p*100:.1f}%")  # show percent with one decimal
    st.progress(min(max(int(round(p*100)), 0), 100))                 # 0–100% progress bar
    st.caption(f"Raw probability: {p:.6f}")                          # exact value so tiny probs aren't 0.00
    st.write(f"**Label (at τ={TAU:.2f}):** {label}")
    st.caption("We always show the raw probability; the label just applies a fixed threshold.")

    # Optional: show the final row used for prediction
    with st.expander("See the full feature values used for this prediction"):
        st.dataframe(pd.DataFrame([used_row], columns=FEATURES).T.rename(columns={0: "value"}))
