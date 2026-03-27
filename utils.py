import pandas as pd
import numpy as np

# ── Column definitions ────────────────────────────────────────────────────────
BINARY_COLS = [
    "owns_kurti", "owns_salwar_suit", "owns_palazzo", "owns_indo_western",
    "owns_night_suit", "owns_bedding_set", "owns_saree", "owns_lehenga",
]

ORDINAL_MAPS = {
    "age_group": {
        "Under_18": 0, "18-24": 1, "25-34": 2,
        "35-44": 3, "45-54": 4, "55+": 5,
    },
    "monthly_income_band": {
        "Below_20K": 1, "20K_40K": 2, "40K_70K": 3,
        "70K_120K": 4, "Above_120K": 5,
    },
    "purchase_frequency": {
        "Rarely": 1, "Festival_Only": 2, "Every_6_Months": 3,
        "Every_2_3_Months": 4, "Monthly_Plus": 5,
    },
    "online_purchase_confidence": {
        "Not_Confident_At_All": 1, "Not_Very_Confident": 2, "Neutral": 3,
        "Fairly_Confident": 4, "Very_Confident": 5,
    },
    "price_sensitivity": {
        "Extremely_Price_Sensitive": 1, "Price_Conscious": 2, "Balanced": 3,
        "Quality_Over_Price": 4, "Price_Irrelevant": 5,
    },
    "brand_openness": {
        "Very_Loyal_One_Brand": 1, "Loyal_But_Open": 2,
        "Multi_Brand_Shopper": 3, "No_Brand_Preference": 4,
    },
    "sustainability_consciousness": {
        "Not_Important": 1, "Neutral": 2,
        "Somewhat_Conscious": 3, "Very_Conscious": 4,
    },
}

CAT_COLS = [
    "region", "city_tier", "occupation", "fashion_identity",
    "preferred_shopping_channel", "discovery_channel",
    "conversion_trigger", "discount_preference",
    "fabric_preference", "color_preference",
]

INC_MID = {
    "Below_20K": 12000, "20K_40K": 30000, "40K_70K": 55000,
    "70K_120K": 95000, "Above_120K": 160000,
}
FREQ_MULT  = {"Monthly_Plus": 12, "Every_2_3_Months": 5, "Every_6_Months": 2,
               "Festival_Only": 2, "Rarely": 1}
PS_FACTOR  = {"Extremely_Price_Sensitive": 0.55, "Price_Conscious": 0.80,
               "Balanced": 1.0, "Quality_Over_Price": 1.30, "Price_Irrelevant": 1.75}

EXPECTED_COLS = [
    "age_group", "region", "city_tier", "occupation",
    "fashion_identity", "price_sensitivity", "brand_openness",
    "online_purchase_confidence", "sustainability_consciousness",
    "purchase_frequency", "preferred_shopping_channel", "discovery_channel",
    "conversion_trigger", "discount_preference",
    "owns_kurti", "owns_salwar_suit", "owns_palazzo", "owns_indo_western",
    "owns_night_suit", "owns_bedding_set", "owns_saree", "owns_lehenga",
    "fabric_preference", "color_preference", "monthly_income_band",
]

# ── Target engineering ────────────────────────────────────────────────────────
def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    bo = d["brand_openness"].map(ORDINAL_MAPS["brand_openness"]).fillna(2)
    oc = d["online_purchase_confidence"].map(ORDINAL_MAPS["online_purchase_confidence"]).fillna(3)
    pf = d["purchase_frequency"].map(ORDINAL_MAPS["purchase_frequency"]).fillna(2)
    ps = d["price_sensitivity"].map(ORDINAL_MAPS["price_sensitivity"]).fillna(3)
    fi_map = {
        "Deeply_Traditional": 1, "Culturally_Rooted": 2, "Comfort_First": 2,
        "Occasion_Driven": 3, "Fusion_Lover": 4, "Trend_First": 4,
    }
    fi = d["fashion_identity"].map(fi_map).fillna(2)
    score = bo * 0.35 + oc * 0.25 + pf * 0.20 + ps * 0.10 + fi * 0.10
    d["zaria_interest_label"] = pd.cut(
        score, bins=[0, 1.9, 2.7, 10],
        labels=["Not_Interested", "Neutral", "Interested"],
    )
    return d

def engineer_spend(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    inc = d["monthly_income_band"].map(INC_MID).fillna(30000)
    fm  = d["purchase_frequency"].map(FREQ_MULT).fillna(2)
    ps  = d["price_sensitivity"].map(PS_FACTOR).fillna(1.0)
    np.random.seed(42)
    noise = np.random.normal(0, 2500, len(d))
    spend = inc * 0.10 * fm * ps + noise
    d["estimated_annual_spend"] = np.clip(spend, 500, 250000).round(-1)
    return d

# ── Feature encoding ──────────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in d.columns:
            d[col + "_enc"] = d[col].map(mapping).fillna(
                pd.Series(mapping.values()).median()
            )
    for col in CAT_COLS:
        if col in d.columns:
            d[col + "_enc"] = pd.Categorical(d[col]).codes.astype(float)
    for col in BINARY_COLS:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)
    return d

def get_feature_matrix(df: pd.DataFrame):
    df_enc = encode_features(df)
    feat_cols = []
    for col in ORDINAL_MAPS:
        c = col + "_enc"
        if c in df_enc.columns:
            feat_cols.append(c)
    for col in CAT_COLS:
        c = col + "_enc"
        if c in df_enc.columns:
            feat_cols.append(c)
    feat_cols += [c for c in BINARY_COLS if c in df_enc.columns]
    X = df_enc[feat_cols].fillna(0)
    return X, feat_cols

# ── Upload validation ─────────────────────────────────────────────────────────
def validate_upload(df: pd.DataFrame):
    errors, warnings = [], []
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    for col in BINARY_COLS:
        if col in df.columns:
            bad = ~df[col].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])
            if bad.any():
                errors.append(f"'{col}' must contain only 0 or 1.")
    if len(df) < 5:
        errors.append("Upload must have at least 5 rows.")
    null_pct = df.isnull().mean() * 100
    for col, pct in null_pct[null_pct > 50].items():
        warnings.append(f"'{col}' has {pct:.0f}% missing values.")
    return errors, warnings

# ── Master loader ─────────────────────────────────────────────────────────────
import streamlit as st

@st.cache_data
def load_data(path: str = "zaria_25col_survey.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = engineer_target(df)
    df = engineer_spend(df)
    return df
