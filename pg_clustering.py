import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
from utils import load_data, encode_features, BINARY_COLS

CLUSTER_NAMES = {
    0: "Festival Splurger",
    1: "Urban Fusion Millennial",
    2: "Daily Cotton Homemaker",
    3: "Premium Occasion Shopper",
    4: "Budget-Conscious Student",
    5: "The Gifter",
}

CLUSTER_OFFERS = {
    "Festival Splurger":          ("Festival Sale 28% off salwar suits (Oct–Nov)",    "WhatsApp + Local retail"),
    "Urban Fusion Millennial":    ("Buy-2-Get-1 Indo-western + Palazzo bundle",        "Instagram Reels + D2C"),
    "Daily Cotton Homemaker":     ("Free shipping on cotton kurtis above ₹799",        "WhatsApp + Word of mouth"),
    "Premium Occasion Shopper":   ("Personalised member offer + early access",         "Email + Brand website"),
    "Budget-Conscious Student":   ("First order ₹150 off + COD available",             "Instagram + Social commerce"),
    "The Gifter":                 ("Gift bundle: kurti + bedding set + gift wrap",     "WhatsApp + Marketplace"),
}

CLUSTER_COLORS = ["#1D9E75","#378ADD","#BA7517","#534AB7","#E24B4A","#888780"]

CLUSTER_FEATS = [
    "fashion_identity_enc","price_sensitivity_enc","brand_openness_enc",
    "online_purchase_confidence_enc","sustainability_consciousness_enc",
    "purchase_frequency_enc",
] + BINARY_COLS

@st.cache_data
def compute_elbow(_df_enc, max_k=10):
    feats = [f for f in CLUSTER_FEATS if f in _df_enc.columns]
    X = _df_enc[feats].fillna(0).values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    inertias, sils = [], []
    for k in range(2, max_k+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, lbl))
    return inertias, sils, Xs

@st.cache_data
def run_kmeans(_Xs, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(_Xs)
    sil    = silhouette_score(_Xs, labels)
    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(_Xs)
    return labels, sil, X_pca

def render():
    st.title("👥 Customer Clustering")
    st.caption("Who are Zaria's distinct customer personas? — K-Means segmentation")

    df = load_data()
    df_enc = encode_features(df)

    with st.spinner("Computing elbow & silhouette curves…"):
        inertias, sils, Xs = compute_elbow(df_enc)

    k_range = list(range(2, 11))

    # ── Elbow & Silhouette ────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Optimal Number of Clusters</div>", unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        fig_el = go.Figure()
        fig_el.add_trace(go.Scatter(x=k_range, y=inertias, mode="lines+markers",
                                    line=dict(color="#1D9E75", width=2),
                                    marker=dict(size=8)))
        fig_el.update_layout(title="Elbow Curve", xaxis_title="k",
                             yaxis_title="Inertia", height=300,
                             margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_el, use_container_width=True)

    with e2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=k_range, y=sils, mode="lines+markers",
                                     line=dict(color="#378ADD", width=2),
                                     marker=dict(size=8)))
        fig_sil.add_vline(x=6, line_dash="dash", line_color="red",
                          annotation_text="k=6 optimal")
        fig_sil.update_layout(title="Silhouette Score", xaxis_title="k",
                              yaxis_title="Silhouette", height=300,
                              margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_sil, use_container_width=True)

    n_clust = st.slider("Number of clusters", 2, 9, 6)

    with st.spinner("Running K-Means…"):
        labels, sil_val, X_pca = run_kmeans(Xs, n_clust)

    df = df.copy()
    df["cluster_id"]   = labels
    df["cluster_name"] = df["cluster_id"].map(
        {i: CLUSTER_NAMES.get(i, f"Cluster {i}") for i in range(n_clust)})

    st.success(f"Silhouette Score: **{sil_val:.3f}** — (higher is better, max = 1.0)")

    # ── PCA Scatter ───────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Cluster Map (PCA 2-D)</div>", unsafe_allow_html=True)
    pca_df = pd.DataFrame({
        "PC1": X_pca[:,0], "PC2": X_pca[:,1],
        "Cluster": df["cluster_name"],
        "Brand Openness": df["brand_openness"],
        "Fashion Identity": df["fashion_identity"],
    })
    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                         color_discrete_sequence=CLUSTER_COLORS[:n_clust],
                         hover_data=["Brand Openness","Fashion Identity"],
                         title="Customer Clusters in 2-D PCA Space", opacity=0.65)
    fig_pca.update_traces(marker=dict(size=5))
    fig_pca.update_layout(height=450, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_pca, use_container_width=True)

    # ── Cluster size ──────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Cluster Size</div>", unsafe_allow_html=True)
    sz = df["cluster_name"].value_counts().reset_index()
    sz.columns = ["Cluster","Count"]
    fig_sz = px.bar(sz, x="Cluster", y="Count", color="Cluster",
                    color_discrete_sequence=CLUSTER_COLORS[:n_clust],
                    text="Count", title="Customers per Cluster")
    fig_sz.update_traces(textposition="outside")
    fig_sz.update_layout(height=320, showlegend=False,
                         margin=dict(t=50,b=10,l=10,r=10), xaxis_title="")
    st.plotly_chart(fig_sz, use_container_width=True)

    # ── Cluster profiles ──────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Cluster Profiles & Marketing Playbook</div>", unsafe_allow_html=True)
    profile_cols = ["price_sensitivity","brand_openness","online_purchase_confidence",
                    "fashion_identity","purchase_frequency","zaria_interest_label"]
    cluster_mode = df.groupby("cluster_name")[profile_cols].agg(
        lambda x: x.mode().iloc[0]).reset_index()

    icons = ["🟢","🔵","🟡","🟣","🔴","⚫"]
    for i, row in cluster_mode.iterrows():
        cn  = row["cluster_name"]
        cnt = (df["cluster_name"] == cn).sum()
        offer, ch = CLUSTER_OFFERS.get(cn, ("Custom offer","Mixed"))
        icon = icons[i % len(icons)]
        with st.expander(f"{icon} **{cn}** — {cnt} customers ({cnt/len(df)*100:.0f}%)"):
            a, b, c_ = st.columns(3)
            a.metric("Price Sensitivity",    row["price_sensitivity"].replace("_"," "))
            b.metric("Brand Openness",        row["brand_openness"].replace("_"," "))
            c_.metric("Purchase Frequency",   row["purchase_frequency"].replace("_"," "))
            d_, e_, f_ = st.columns(3)
            d_.metric("Fashion Identity",     row["fashion_identity"].replace("_"," "))
            e_.metric("Online Confidence",    row["online_purchase_confidence"].replace("_"," "))
            f_.metric("Interest Label",       str(row["zaria_interest_label"]))
            st.markdown(f"**Recommended Offer →** {offer}")
            st.markdown(f"**Best Channel →** {ch}")

    # ── Discount preference per cluster ───────────────────────────────────────
    st.markdown("<div class='sec'>Discount Strategy per Cluster</div>", unsafe_allow_html=True)
    disc = (df.groupby("cluster_name")["discount_preference"]
              .agg(lambda x: x.value_counts().idxmax()).reset_index())
    disc.columns = ["Cluster","Top Discount Preference"]
    for _, row in disc.iterrows():
        offer, ch = CLUSTER_OFFERS.get(row["Cluster"],("—","—"))
        st.markdown(
            f"**{row['Cluster']}** → preferred discount: `{row['Top Discount Preference'].replace('_',' ')}`"
            f" | Zaria offer: _{offer}_ | Channel: _{ch}_"
        )
