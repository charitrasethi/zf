import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data

def render():
    st.title("🏠 Executive Overview")
    st.caption("Founder's morning brief — 60-second market snapshot")

    df = load_data()

    interested  = (df["zaria_interest_label"] == "Interested").sum()
    neutral     = (df["zaria_interest_label"] == "Neutral").sum()
    not_int     = (df["zaria_interest_label"] == "Not_Interested").sum()
    avg_spend   = df["estimated_annual_spend"].mean()
    top_region  = df["region"].value_counts().idxmax().replace("_", " ")
    top_trigger = df["conversion_trigger"].value_counts().idxmax().replace("_", " ")

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (len(df), "Total Respondents", "#2C2C2A"),
        (f"{interested:,}", "Interested in Zaria", "#1D9E75"),
        (f"{neutral:,}", "Neutral", "#FFC107"),
        (f"{not_int:,}", "Not Interested", "#E24B4A"),
        (f"₹{avg_spend/1000:.1f}K", "Avg Annual Spend", "#185FA5"),
        (f"{interested/len(df)*100:.0f}%", "Interest Rate", "#534AB7"),
    ]
    for col, (val, lbl, color) in zip([c1,c2,c3,c4,c5,c6], kpis):
        with col:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-val' style='color:{color}'>{val}</div>"
                f"<div class='kpi-lbl'>{lbl}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        st.markdown("<div class='sec'>Interest Distribution</div>", unsafe_allow_html=True)
        fig = px.pie(
            names=["Interested", "Neutral", "Not Interested"],
            values=[interested, neutral, not_int],
            color_discrete_sequence=["#1D9E75", "#FFC107", "#E24B4A"],
            hole=0.45,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, height=280,
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown("<div class='sec'>Interested by Region</div>", unsafe_allow_html=True)
        rdf = (df[df["zaria_interest_label"] == "Interested"]["region"]
               .value_counts().reset_index())
        rdf.columns = ["Region", "Count"]
        rdf["Region"] = rdf["Region"].str.replace("_", " ")
        fig2 = px.bar(rdf, x="Count", y="Region", orientation="h",
                      color="Count", color_continuous_scale=["#9FE1CB", "#085041"],
                      text="Count")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=280, coloraxis_showscale=False,
                           margin=dict(t=10, b=10, l=10, r=10),
                           yaxis_title="", xaxis_title="Interested Customers")
        st.plotly_chart(fig2, use_container_width=True)

    with r1c3:
        st.markdown("<div class='sec'>City Tier Breakdown</div>", unsafe_allow_html=True)
        tdf = df["city_tier"].value_counts().reset_index()
        tdf.columns = ["Tier", "Count"]
        fig3 = px.bar(tdf, x="Tier", y="Count",
                      color="Tier",
                      color_discrete_sequence=["#1D9E75", "#378ADD", "#BA7517", "#888780"],
                      text="Count")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=280, showlegend=False,
                           margin=dict(t=10, b=10, l=10, r=10),
                           xaxis_title="", yaxis_title="Respondents")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Row 2 ─────────────────────────────────────────────────────────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("<div class='sec'>Fashion Identity</div>", unsafe_allow_html=True)
        fidf = df["fashion_identity"].value_counts().reset_index()
        fidf.columns = ["Identity", "Count"]
        fidf["Identity"] = fidf["Identity"].str.replace("_", " ")
        fig4 = px.bar(fidf, x="Identity", y="Count",
                      color="Count", color_continuous_scale=["#CECBF6", "#26215C"],
                      text="Count")
        fig4.update_traces(textposition="outside")
        fig4.update_layout(height=300, coloraxis_showscale=False,
                           margin=dict(t=10, b=10, l=10, r=10),
                           xaxis_title="", xaxis_tickangle=-15)
        st.plotly_chart(fig4, use_container_width=True)

    with r2c2:
        st.markdown("<div class='sec'>Top Conversion Triggers</div>", unsafe_allow_html=True)
        ctdf = df["conversion_trigger"].value_counts().reset_index()
        ctdf.columns = ["Trigger", "Count"]
        ctdf["Trigger"] = ctdf["Trigger"].str.replace("_", " ")
        fig5 = px.bar(ctdf, x="Count", y="Trigger", orientation="h",
                      color="Count", color_continuous_scale=["#B5D4F4", "#042C53"],
                      text="Count")
        fig5.update_traces(textposition="outside")
        fig5.update_layout(height=300, coloraxis_showscale=False,
                           margin=dict(t=10, b=10, l=10, r=10),
                           yaxis_title="", xaxis_title="Count")
        st.plotly_chart(fig5, use_container_width=True)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='sec'>Founder's Key Insights</div>", unsafe_allow_html=True)
    top_channel = df["preferred_shopping_channel"].value_counts().idxmax().replace("_", " ")
    top_inc     = df["monthly_income_band"].value_counts().idxmax().replace("_", " ")

    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.markdown(f"<div class='ibox'>🎯 <b>{interested/len(df)*100:.0f}%</b> of respondents are interested — strong product-market fit for a new brand.</div>", unsafe_allow_html=True)
    with ic2:
        st.markdown(f"<div class='ibox'>🔑 Top trigger: <b>{top_trigger}</b> — lead with this in launch messaging.</div>", unsafe_allow_html=True)
    with ic3:
        st.markdown(f"<div class='ibox'>🛒 Preferred channel: <b>{top_channel}</b> — invest here first.</div>", unsafe_allow_html=True)
    with ic4:
        st.markdown(f"<div class='ibox'>💰 Dominant income band: <b>{top_inc}</b> — price ₹800–₹2,500 for widest reach.</div>", unsafe_allow_html=True)
