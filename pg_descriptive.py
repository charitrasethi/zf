import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, BINARY_COLS

def render():
    st.title("📊 Descriptive Analysis")
    st.caption("What is the current state of Zaria's potential market?")

    df = load_data()

    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        sel_region = st.multiselect("Region", sorted(df["region"].unique()),
                                     default=list(df["region"].unique()))
        sel_tier   = st.multiselect("City Tier", sorted(df["city_tier"].unique()),
                                     default=list(df["city_tier"].unique()))
        sel_age    = st.multiselect("Age Group", sorted(df["age_group"].unique()),
                                     default=list(df["age_group"].unique()))

    df = df[df["region"].isin(sel_region) &
            df["city_tier"].isin(sel_tier) &
            df["age_group"].isin(sel_age)].copy()
    st.caption(f"Showing **{len(df):,}** respondents after filters")

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Demographics</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)

    with d1:
        age_order = ["Under_18","18-24","25-34","35-44","45-54","55+"]
        adf = df["age_group"].value_counts().reindex(age_order, fill_value=0).reset_index()
        adf.columns = ["Age","Count"]
        fig = px.bar(adf, x="Age", y="Count", text="Count",
                     color="Count", color_continuous_scale=["#9FE1CB","#085041"],
                     title="Age Distribution")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, coloraxis_showscale=False,
                          margin=dict(t=40,b=10,l=5,r=5), xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        odf = df["occupation"].value_counts().reset_index()
        odf.columns = ["Occupation","Count"]
        odf["Occupation"] = odf["Occupation"].str.replace("_"," ")
        fig2 = px.pie(odf, names="Occupation", values="Count", hole=0.38,
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      title="Occupation Mix")
        fig2.update_layout(height=300, margin=dict(t=40,b=10,l=5,r=5))
        st.plotly_chart(fig2, use_container_width=True)

    with d3:
        inc_order = ["Below_20K","20K_40K","40K_70K","70K_120K","Above_120K"]
        idf = df["monthly_income_band"].value_counts().reindex(inc_order, fill_value=0).reset_index()
        idf.columns = ["Income","Count"]
        fig3 = px.bar(idf, x="Income", y="Count", text="Count",
                      color="Count", color_continuous_scale=["#B5D4F4","#042C53"],
                      title="Income Distribution")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=300, coloraxis_showscale=False,
                           margin=dict(t=40,b=10,l=5,r=5),
                           xaxis_title="", xaxis_tickangle=-15)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Psychographics ────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Psychographic Profiles</div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)

    with p1:
        ps_order = ["Extremely_Price_Sensitive","Price_Conscious","Balanced",
                    "Quality_Over_Price","Price_Irrelevant"]
        pdf = df["price_sensitivity"].value_counts().reindex(ps_order, fill_value=0).reset_index()
        pdf.columns = ["Sensitivity","Count"]
        pdf["Sensitivity"] = pdf["Sensitivity"].str.replace("_"," ")
        fig4 = px.bar(pdf, x="Count", y="Sensitivity", orientation="h",
                      color="Count", color_continuous_scale=["#FAC775","#412402"],
                      text="Count", title="Price Sensitivity")
        fig4.update_traces(textposition="outside")
        fig4.update_layout(height=300, coloraxis_showscale=False,
                           margin=dict(t=40,b=10,l=5,r=5), yaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    with p2:
        bo_order = ["Very_Loyal_One_Brand","Loyal_But_Open",
                    "Multi_Brand_Shopper","No_Brand_Preference"]
        bdf = df["brand_openness"].value_counts().reindex(bo_order, fill_value=0).reset_index()
        bdf.columns = ["Openness","Count"]
        bdf["Openness"] = bdf["Openness"].str.replace("_"," ")
        fig5 = px.funnel(bdf, x="Count", y="Openness",
                         color_discrete_sequence=["#1D9E75"],
                         title="Brand Openness Funnel")
        fig5.update_layout(height=300, margin=dict(t=40,b=10,l=5,r=5))
        st.plotly_chart(fig5, use_container_width=True)

    with p3:
        sc_df = df["sustainability_consciousness"].value_counts().reset_index()
        sc_df.columns = ["Level","Count"]
        sc_df["Level"] = sc_df["Level"].str.replace("_"," ")
        fig6 = px.pie(sc_df, names="Level", values="Count", hole=0.38,
                      color_discrete_sequence=["#173404","#3B6D11","#97C459","#C0DD97"],
                      title="Sustainability Consciousness")
        fig6.update_layout(height=300, margin=dict(t=40,b=10,l=5,r=5))
        st.plotly_chart(fig6, use_container_width=True)

    # ── Product Ownership Heatmap ─────────────────────────────────────────────
    st.markdown("<div class='sec'>Product Ownership by Region (%)</div>", unsafe_allow_html=True)
    heat = df.groupby("region")[BINARY_COLS].mean().round(3) * 100
    heat.index = heat.index.str.replace("_"," ")
    heat.columns = [c.replace("owns_","").replace("_"," ").title() for c in heat.columns]
    fig7 = px.imshow(heat, color_continuous_scale="Greens", text_auto=".0f",
                     aspect="auto", labels=dict(color="Ownership %"),
                     title="% Customers owning each product — by Region")
    fig7.update_layout(height=340, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig7, use_container_width=True)

    # ── Shopping Behaviour ────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Shopping Behaviour</div>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    with s1:
        pf_order = ["Monthly_Plus","Every_2_3_Months","Every_6_Months","Festival_Only","Rarely"]
        pfdf = df["purchase_frequency"].value_counts().reindex(pf_order, fill_value=0).reset_index()
        pfdf.columns = ["Frequency","Count"]
        pfdf["Frequency"] = pfdf["Frequency"].str.replace("_"," ")
        fig8 = px.bar(pfdf, x="Frequency", y="Count", text="Count",
                      color="Count", color_continuous_scale=["#9FE1CB","#085041"],
                      title="Purchase Frequency")
        fig8.update_traces(textposition="outside")
        fig8.update_layout(height=300, coloraxis_showscale=False,
                           margin=dict(t=40,b=10,l=5,r=5),
                           xaxis_title="", xaxis_tickangle=-15)
        st.plotly_chart(fig8, use_container_width=True)

    with s2:
        chdf = df["preferred_shopping_channel"].value_counts().reset_index()
        chdf.columns = ["Channel","Count"]
        chdf["Channel"] = chdf["Channel"].str.replace("_"," ")
        fig9 = px.pie(chdf, names="Channel", values="Count", hole=0.35,
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      title="Preferred Channel")
        fig9.update_layout(height=300, margin=dict(t=40,b=10,l=5,r=5))
        st.plotly_chart(fig9, use_container_width=True)

    with s3:
        dcdf = df["discovery_channel"].value_counts().reset_index()
        dcdf.columns = ["Channel","Count"]
        dcdf["Channel"] = dcdf["Channel"].str.replace("_"," ")
        fig10 = px.bar(dcdf, x="Count", y="Channel", orientation="h",
                       color="Count", color_continuous_scale=["#CECBF6","#26215C"],
                       text="Count", title="Discovery Channel")
        fig10.update_traces(textposition="outside")
        fig10.update_layout(height=300, coloraxis_showscale=False,
                            margin=dict(t=40,b=10,l=5,r=5), yaxis_title="")
        st.plotly_chart(fig10, use_container_width=True)

    # ── Fabric & Colour ───────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Fabric & Colour Preferences</div>", unsafe_allow_html=True)
    f1, f2 = st.columns(2)

    with f1:
        fabdf = df["fabric_preference"].value_counts().reset_index()
        fabdf.columns = ["Fabric","Count"]
        fabdf["Fabric"] = fabdf["Fabric"].str.replace("_"," ")
        fig11 = px.bar(fabdf, x="Fabric", y="Count", text="Count",
                       color="Fabric",
                       color_discrete_sequence=px.colors.qualitative.Safe,
                       title="Fabric Preference")
        fig11.update_traces(textposition="outside")
        fig11.update_layout(height=300, showlegend=False,
                            margin=dict(t=40,b=10,l=5,r=5), xaxis_title="")
        st.plotly_chart(fig11, use_container_width=True)

    with f2:
        coldf = df["color_preference"].value_counts().reset_index()
        coldf.columns = ["Color","Count"]
        coldf["Color"] = coldf["Color"].str.replace("_"," ")
        pal = {"Pastels":"#F4C0D1","Bright Vibrant":"#E24B4A",
               "Jewel Tones":"#534AB7","Neutrals":"#B4B2A9",
               "Dark Tones":"#2C2C2A","Bold Prints":"#BA7517"}
        fig12 = px.bar(coldf, x="Color", y="Count", text="Count",
                       color="Color", color_discrete_map=pal,
                       title="Colour Preference")
        fig12.update_traces(textposition="outside")
        fig12.update_layout(height=300, showlegend=False,
                            margin=dict(t=40,b=10,l=5,r=5), xaxis_title="")
        st.plotly_chart(fig12, use_container_width=True)

    with st.expander("📋 Raw Data (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button("Download filtered CSV",
                           df.to_csv(index=False).encode(),
                           "zaria_filtered.csv", "text/csv")
