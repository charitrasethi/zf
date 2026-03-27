import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")
from utils import load_data, BINARY_COLS

def build_basket(df: pd.DataFrame) -> pd.DataFrame:
    basket = df[BINARY_COLS].copy()
    # Add fabric one-hot
    for fab in df["fabric_preference"].dropna().unique():
        basket[f"Fabric_{fab}"] = (df["fabric_preference"] == fab).astype(int)
    # Add color one-hot
    for col in df["color_preference"].dropna().unique():
        basket[f"Color_{col}"] = (df["color_preference"] == col).astype(int)
    basket.columns = [
        c.replace("owns_","").replace("_"," ").title() for c in basket.columns
    ]
    return basket.astype(bool)

@st.cache_data
def run_arm(min_sup: float, min_conf: float, region: str):
    df = load_data()
    if region != "All":
        df = df[df["region"] == region]
    if len(df) < 20:
        return pd.DataFrame(), 0

    basket  = build_basket(df)
    freq    = apriori(basket, min_support=min_sup, use_colnames=True)
    if freq.empty:
        return pd.DataFrame(), 0

    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules[rules["lift"] >= 1.0].copy()
    rules.sort_values("lift", ascending=False, inplace=True)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: " + ".join(sorted(x)))
    out = rules[["antecedents_str","consequents_str","support","confidence","lift"]].round(4)
    return out, len(df)

def render():
    st.title("🔗 Association Rule Mining")
    st.caption("What do customers own together? — Apriori with Support · Confidence · Lift")

    df = load_data()

    # Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_sup  = st.slider("Min Support",    0.03, 0.30, 0.07, 0.01)
    with c2:
        min_conf = st.slider("Min Confidence", 0.30, 0.90, 0.50, 0.05)
    with c3:
        min_lift = st.slider("Min Lift",       1.0,  3.0,  1.1,  0.1)
    with c4:
        region_f = st.selectbox("Region", ["All"] + sorted(df["region"].unique()))

    with st.spinner("Mining association rules…"):
        rules_raw, n_rows = run_arm(min_sup, min_conf, region_f)

    if rules_raw.empty:
        st.warning("No rules found. Try lowering Min Support or Min Confidence.")
        return

    rules = rules_raw[rules_raw["lift"] >= min_lift].copy()
    st.success(
        f"**{len(rules)}** rules found on **{n_rows}** respondents "
        f"(support ≥ {min_sup}, confidence ≥ {min_conf}, lift ≥ {min_lift})"
    )

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Rules",     len(rules))
    k2.metric("Max Lift",        f"{rules['lift'].max():.2f}")
    k3.metric("Avg Confidence",  f"{rules['confidence'].mean():.2%}")
    k4.metric("Avg Support",     f"{rules['support'].mean():.3f}")

    # ── Rules table ───────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Top Rules (sorted by Lift)</div>", unsafe_allow_html=True)
    disp = rules.head(30).copy()
    disp.columns = ["If customer has →", "→ They also have",
                    "Support", "Confidence", "Lift"]
    disp["Support"]    = disp["Support"].map("{:.3f}".format)
    disp["Confidence"] = disp["Confidence"].map("{:.3f}".format)
    disp["Lift"]       = disp["Lift"].map("{:.2f}".format)
    st.dataframe(disp, use_container_width=True, height=380)
    st.download_button("Download Rules CSV",
                       rules.to_csv(index=False).encode(),
                       "zaria_arm_rules.csv", "text/csv")

    # ── Support vs Confidence scatter (size = lift) ───────────────────────────
    st.markdown("<div class='sec'>Support × Confidence × Lift</div>", unsafe_allow_html=True)
    rules["rule_label"] = rules["antecedents_str"] + " → " + rules["consequents_str"]
    fig_sc = px.scatter(
        rules, x="support", y="confidence", size="lift", color="lift",
        color_continuous_scale="Greens",
        hover_data=["rule_label","lift"],
        title="Support vs Confidence (bubble size = Lift)",
        labels={"support":"Support","confidence":"Confidence","lift":"Lift"},
    )
    fig_sc.update_layout(height=420, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Lift & Confidence distributions ──────────────────────────────────────
    st.markdown("<div class='sec'>Lift & Confidence Distributions</div>", unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        fig_lh = px.histogram(rules, x="lift", nbins=20,
                              color_discrete_sequence=["#1D9E75"],
                              title="Lift Distribution",
                              labels={"lift":"Lift","count":"Rules"})
        fig_lh.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_lh, use_container_width=True)

    with h2:
        fig_ch = px.histogram(rules, x="confidence", nbins=20,
                              color_discrete_sequence=["#378ADD"],
                              title="Confidence Distribution",
                              labels={"confidence":"Confidence","count":"Rules"})
        fig_ch.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_ch, use_container_width=True)

    # ── Top antecedents bar ───────────────────────────────────────────────────
    st.markdown("<div class='sec'>Most Frequent Antecedents</div>", unsafe_allow_html=True)
    ant_counts = rules["antecedents_str"].value_counts().head(12).reset_index()
    ant_counts.columns = ["Antecedent","Count"]
    fig_ant = px.bar(ant_counts, x="Count", y="Antecedent", orientation="h",
                     color="Count", color_continuous_scale=["#9FE1CB","#085041"],
                     text="Count", title="Most common rule antecedents (LHS)")
    fig_ant.update_traces(textposition="outside")
    fig_ant.update_layout(height=380, coloraxis_showscale=False,
                          margin=dict(t=50,b=10,l=10,r=10), yaxis_title="")
    st.plotly_chart(fig_ant, use_container_width=True)

    # ── Network graph ─────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Association Network Graph (top 25 rules)</div>", unsafe_allow_html=True)
    top25 = rules.head(25)
    G = nx.DiGraph()
    for _, row in top25.iterrows():
        G.add_edge(row["antecedents_str"], row["consequents_str"],
                   weight=row["confidence"], lift=row["lift"])

    pos = nx.spring_layout(G, seed=42, k=2.5)
    ex, ey = [], []
    for u, v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        ex += [x0,x1,None]; ey += [y0,y1,None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_sz = [max(16, G.degree(n)*9) for n in G.nodes()]
    node_lbl = list(G.nodes())

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                 line=dict(width=1.2, color="#aaaaaa"),
                                 hoverinfo="none"))
    fig_net.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                 marker=dict(size=node_sz, color="#1D9E75",
                                             line=dict(color="#fff",width=1.5)),
                                 text=node_lbl, textposition="top center",
                                 textfont=dict(size=9), hoverinfo="text"))
    fig_net.update_layout(
        showlegend=False, height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=10,b=10,l=10,r=10),
        title="Product Association Network",
    )
    st.plotly_chart(fig_net, use_container_width=True)

    # ── Bundle recommendations ────────────────────────────────────────────────
    st.markdown("<div class='sec'>🛍️ Bundle Recommendations (high-lift rules)</div>", unsafe_allow_html=True)
    q75 = rules["lift"].quantile(0.75)
    bundles = rules[rules["lift"] >= q75].head(6)
    if not bundles.empty:
        for _, r in bundles.iterrows():
            st.markdown(
                f"📦 **{r['antecedents_str']}** + **{r['consequents_str']}** "
                f"| Confidence: `{float(r['confidence']):.0%}` "
                f"| Lift: `{float(r['lift']):.2f}×`"
            )
    else:
        st.info("Lower lift threshold to see bundle suggestions.")

    # ── Regional comparison ───────────────────────────────────────────────────
    st.markdown("<div class='sec'>Regional Rule Count Comparison</div>", unsafe_allow_html=True)
    regions = df["region"].unique().tolist()
    reg_counts = {}
    for reg in regions:
        r2, _ = run_arm(min_sup, min_conf, reg)
        reg_counts[reg] = int(len(r2[r2["lift"] >= min_lift])) if not r2.empty else 0

    rc_df = pd.DataFrame(list(reg_counts.items()), columns=["Region","Rules"])
    rc_df["Region"] = rc_df["Region"].str.replace("_"," ")
    rc_df = rc_df.sort_values("Rules", ascending=False)
    fig_rc = px.bar(rc_df, x="Region", y="Rules",
                    color="Rules", color_continuous_scale=["#9FE1CB","#085041"],
                    text="Rules", title="Valid Rules per Region")
    fig_rc.update_traces(textposition="outside")
    fig_rc.update_layout(height=300, coloraxis_showscale=False,
                         margin=dict(t=50,b=10,l=10,r=10), xaxis_title="")
    st.plotly_chart(fig_rc, use_container_width=True)
    st.markdown("<div class='ibox'>📌 Regions with more rules = richer co-purchase patterns = better bundle targeting opportunities in that geography.</div>", unsafe_allow_html=True)
