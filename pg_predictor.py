import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import (
    load_data, encode_features, get_feature_matrix,
    validate_upload, engineer_target, engineer_spend,
    BINARY_COLS, EXPECTED_COLS,
)

CLUSTER_NAMES = {
    0:"Festival Splurger", 1:"Urban Fusion Millennial",
    2:"Daily Cotton Homemaker", 3:"Premium Occasion Shopper",
    4:"Budget-Conscious Student", 5:"The Gifter",
}
CLUSTER_OFFERS = {
    "Festival Splurger":        ("Festival Sale 28% off salwar suits","WhatsApp + Local retail"),
    "Urban Fusion Millennial":  ("Buy2Get1 Indo-western + Palazzo bundle","Instagram Reels + D2C"),
    "Daily Cotton Homemaker":   ("Free shipping on cotton kurtis > ₹799","WhatsApp + Word of mouth"),
    "Premium Occasion Shopper": ("Personalised offer + early collection access","Email + Brand website"),
    "Budget-Conscious Student": ("First order ₹150 off + COD available","Instagram + Social commerce"),
    "The Gifter":               ("Gift bundle: kurti + bedding + gift wrap","WhatsApp + Marketplace"),
}

# ── Cluster assignment (re-fit on base data, predict on new) ─────────────────
@st.cache_resource
def get_cluster_model():
    df     = load_data()
    df_enc = encode_features(df)
    feats  = ["fashion_identity_enc","price_sensitivity_enc","brand_openness_enc",
              "online_purchase_confidence_enc","sustainability_consciousness_enc",
              "purchase_frequency_enc"] + BINARY_COLS
    feats  = [f for f in feats if f in df_enc.columns]
    X      = df_enc[feats].fillna(0).values
    sc     = StandardScaler()
    Xs     = sc.fit_transform(X)
    km     = KMeans(n_clusters=6, random_state=42, n_init=10)
    km.fit(Xs)
    return km, sc, feats

def predict_cluster(df_new):
    km, sc, feats = get_cluster_model()
    df_enc = encode_features(df_new)
    X_new  = np.zeros((len(df_new), len(feats)))
    for i, f in enumerate(feats):
        if f in df_enc.columns:
            X_new[:, i] = df_enc[f].fillna(0).values
    labels = km.predict(sc.transform(X_new))
    return [CLUSTER_NAMES.get(l, f"Cluster {l}") for l in labels]

# ── Template ──────────────────────────────────────────────────────────────────
TEMPLATE_ROW = {
    "age_group":"25-34","region":"North_India","city_tier":"Metro",
    "occupation":"Salaried_Private","fashion_identity":"Fusion_Lover",
    "price_sensitivity":"Balanced","brand_openness":"Multi_Brand_Shopper",
    "online_purchase_confidence":"Fairly_Confident",
    "sustainability_consciousness":"Somewhat_Conscious",
    "purchase_frequency":"Every_2_3_Months",
    "preferred_shopping_channel":"Online_Marketplace",
    "discovery_channel":"Instagram_Reels",
    "conversion_trigger":"Customer_Reviews",
    "discount_preference":"Buy2_Get1_Bundle",
    "owns_kurti":1,"owns_salwar_suit":1,"owns_palazzo":0,
    "owns_indo_western":1,"owns_night_suit":0,"owns_bedding_set":0,
    "owns_saree":0,"owns_lehenga":0,
    "fabric_preference":"Pure_Cotton","color_preference":"Pastels",
    "monthly_income_band":"40K_70K",
}

def render():
    st.title("🆕 New Customer Predictor")
    st.caption("Upload new survey responses → instant predictions · cluster assignment · personalised offer")

    clf_ok = os.path.exists("zaria_clf.pkl")
    reg_ok = os.path.exists("zaria_reg.pkl")

    if not clf_ok or not reg_ok:
        st.warning(
            "⚠️ Trained models not found. "
            "Visit **Classification** and **Regression & CLV** pages first — "
            "models are auto-saved when you open those pages."
        )

    # ── Template download ─────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Step 1 — Download the template</div>", unsafe_allow_html=True)
    tpl = pd.DataFrame([TEMPLATE_ROW])
    st.download_button("📥 Download CSV Template (25 columns)",
                       tpl.to_csv(index=False).encode(),
                       "zaria_template.csv","text/csv")
    st.caption("Fill the template with your new survey responses and upload below.")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Step 2 — Upload your data</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (25 survey columns)", type=["csv"])

    if uploaded is None:
        # Demo preview
        st.info("👆 Upload a CSV to get predictions. Preview of output format:")
        demo = pd.DataFrame({
            "interest_probability": [0.87, 0.43, 0.21],
            "predicted_interest":   ["Interested","Neutral","Not_Interested"],
            "lead_priority":        ["Hot Lead","Nurture","Low Priority"],
            "cluster_assigned":     ["Urban Fusion Millennial","Festival Splurger","Budget-Conscious Student"],
            "pred_annual_spend ₹":  [34500, 12000, 4200],
            "recommended_offer":    ["Buy2Get1 Indo-western bundle","Festival Sale 28% off","₹150 off first order"],
            "recommended_channel":  ["Instagram Reels","WhatsApp","Social Commerce"],
        })
        st.dataframe(demo, use_container_width=True)
        return

    try:
        df_new = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Cannot read file: {e}")
        return

    # ── Validate ──────────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Step 3 — Validation</div>", unsafe_allow_html=True)
    errors, warns = validate_upload(df_new)
    for e in errors:
        st.error(f"❌ {e}")
    if errors:
        return
    st.success(f"✅ {len(df_new)} rows · {len(df_new.columns)} columns — validation passed")
    for w in warns:
        st.warning(f"⚠️ {w}")

    with st.expander("Preview uploaded data"):
        st.dataframe(df_new.head(10), use_container_width=True)

    # ── Predict ───────────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Step 4 — Predictions</div>", unsafe_allow_html=True)
    out = df_new.copy()

    with st.spinner("Running models…"):
        # Classification
        if clf_ok:
            bundle = joblib.load("zaria_clf.pkl")
            clf_model, le_clf, feat_cols_clf = (
                bundle["model"], bundle["le"], bundle["feat_cols"])
            df_enc = encode_features(df_new)
            X_clf  = np.zeros((len(df_new), len(feat_cols_clf)))
            for i, c in enumerate(feat_cols_clf):
                if c in df_enc.columns:
                    X_clf[:, i] = df_enc[c].fillna(0).values
            probs    = clf_model.predict_proba(X_clf)
            preds    = clf_model.predict(X_clf)
            int_idx  = list(le_clf.classes_).index("Interested") \
                       if "Interested" in le_clf.classes_ else 0
            out["interest_probability"] = probs[:, int_idx].round(3)
            out["predicted_interest"]   = le_clf.inverse_transform(preds)
            out["lead_priority"]        = pd.cut(
                out["interest_probability"],
                bins=[0, 0.40, 0.70, 1.01],
                labels=["Low Priority","Nurture","Hot Lead"],
            )
        else:
            out["interest_probability"] = 0.5
            out["predicted_interest"]   = "Unknown"
            out["lead_priority"]        = "Unknown"

        # Regression
        if reg_ok:
            rb = joblib.load("zaria_reg.pkl")
            reg_model, feat_cols_reg = rb["model"], rb["feat_cols"]
            df_enc2 = encode_features(df_new)
            X_reg   = np.zeros((len(df_new), len(feat_cols_reg)))
            for i, c in enumerate(feat_cols_reg):
                if c in df_enc2.columns:
                    X_reg[:, i] = df_enc2[c].fillna(0).values
            out["pred_annual_spend"] = np.expm1(reg_model.predict(X_reg)).round(-1)
        else:
            df_proc = engineer_spend(engineer_target(df_new.copy()))
            out["pred_annual_spend"] = df_proc["estimated_annual_spend"]

        # Clustering
        out["cluster_assigned"]   = predict_cluster(df_new)
        out["recommended_offer"]  = out["cluster_assigned"].map(
            lambda x: CLUSTER_OFFERS.get(x,("Custom offer","—"))[0])
        out["recommended_channel"]= out["cluster_assigned"].map(
            lambda x: CLUSTER_OFFERS.get(x,("—","Mixed"))[1])
        if "conversion_trigger" in df_new.columns:
            out["key_trigger"] = df_new["conversion_trigger"]

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    hot  = (out.get("lead_priority","") == "Hot Lead").sum()
    nurt = (out.get("lead_priority","") == "Nurture").sum()
    low  = (out.get("lead_priority","") == "Low Priority").sum()
    avg_s= out["pred_annual_spend"].mean() if "pred_annual_spend" in out else 0

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("🔥 Hot Leads",    int(hot))
    k2.metric("🌱 Nurture",      int(nurt))
    k3.metric("📉 Low Priority", int(low))
    k4.metric("💰 Avg Pred Spend",f"₹{avg_s:,.0f}")

    # Lead priority pie
    p1, p2 = st.columns(2)
    with p1:
        pri = out["lead_priority"].value_counts().reset_index()
        pri.columns = ["Priority","Count"]
        fig_p = px.pie(pri, names="Priority", values="Count",
                       color_discrete_sequence=["#1D9E75","#FFC107","#E24B4A"],
                       title="Lead Priority Distribution", hole=0.4)
        fig_p.update_layout(height=300, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_p, use_container_width=True)

    with p2:
        cl = out["cluster_assigned"].value_counts().reset_index()
        cl.columns = ["Cluster","Count"]
        fig_c = px.bar(cl, x="Cluster", y="Count", color="Cluster",
                       color_discrete_sequence=["#1D9E75","#378ADD","#BA7517",
                                                 "#534AB7","#E24B4A","#888780"],
                       text="Count", title="Cluster Distribution")
        fig_c.update_traces(textposition="outside")
        fig_c.update_layout(height=300, showlegend=False,
                            margin=dict(t=50,b=10,l=10,r=10), xaxis_title="")
        st.plotly_chart(fig_c, use_container_width=True)

    # Hot lead cards
    hot_rows = out[out["lead_priority"]=="Hot Lead"].head(5)
    if not hot_rows.empty:
        st.markdown("<div class='sec'>🔥 Hot Leads — Act Now</div>", unsafe_allow_html=True)
        for idx, row in hot_rows.iterrows():
            with st.expander(
                f"Customer #{idx+1} — Probability: {row.get('interest_probability',0):.0%} "
                f"| Cluster: {row.get('cluster_assigned','—')}"
            ):
                a, b = st.columns(2)
                a.metric("Interest Probability", f"{row.get('interest_probability',0):.0%}")
                b.metric("Predicted Annual Spend", f"₹{row.get('pred_annual_spend',0):,.0f}")
                st.markdown(f"**Cluster:** {row.get('cluster_assigned','—')}")
                st.markdown(f"**Recommended Offer:** {row.get('recommended_offer','—')}")
                st.markdown(f"**Best Channel:** {row.get('recommended_channel','—')}")
                if "key_trigger" in row:
                    st.markdown(f"**Conversion Trigger:** {str(row['key_trigger']).replace('_',' ')}")

    # Full table
    st.markdown("<div class='sec'>Full Results Table</div>", unsafe_allow_html=True)
    pred_cols = ["interest_probability","predicted_interest","lead_priority",
                 "cluster_assigned","pred_annual_spend",
                 "recommended_offer","recommended_channel"]
    pred_cols = [c for c in pred_cols if c in out.columns]
    st.dataframe(out[pred_cols].reset_index(drop=True), use_container_width=True)

    # Enriched download
    all_cols = list(df_new.columns) + [c for c in pred_cols if c not in df_new.columns]
    st.download_button(
        "📥 Download Enriched Predictions CSV",
        out[all_cols].to_csv(index=False).encode(),
        "zaria_predictions.csv","text/csv",
    )
    st.markdown("<div class='ibox'>✅ The enriched CSV contains all 25 original columns + interest probability, lead priority, cluster, predicted spend, recommended offer & channel. Import directly into your CRM or WhatsApp broadcast tool.</div>", unsafe_allow_html=True)
