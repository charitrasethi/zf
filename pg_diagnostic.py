import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data, encode_features, ORDINAL_MAPS

def render():
    st.title("🔍 Diagnostic Analysis")
    st.caption("Why is it happening? Correlations, cross-tabs, causal patterns.")

    df = load_data()
    df_enc = encode_features(df)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown("<div class='sec'>Psychographic Correlation Heatmap</div>", unsafe_allow_html=True)

    corr_cols = [c + "_enc" for c in ORDINAL_MAPS if c + "_enc" in df_enc.columns]
    corr_labels = [c.replace("_enc","").replace("_"," ").title() for c in corr_cols]
    corr_mat = df_enc[corr_cols].corr().round(2)
    corr_mat.index = corr_labels
    corr_mat.columns = corr_labels

    fig_h, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9})
    ax.set_title("Correlation Matrix — Ordinal Features", fontsize=12, pad=8)
    plt.tight_layout()
    st.pyplot(fig_h, use_container_width=True)
    plt.close()

    st.markdown("<div class='ibox'>📌 Income, price sensitivity, and brand openness show the strongest correlation cluster — these three jointly predict customer value.</div>", unsafe_allow_html=True)

    # ── Cross-tab explorer ────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Interactive Cross-Tab Explorer</div>", unsafe_allow_html=True)
    cat_cols_all = [
        "region","city_tier","occupation","fashion_identity","price_sensitivity",
        "brand_openness","online_purchase_confidence","discovery_channel",
        "conversion_trigger","discount_preference","fabric_preference",
        "color_preference","purchase_frequency","zaria_interest_label",
    ]
    cx1, cx2 = st.columns(2)
    with cx1:
        x_col = st.selectbox("Row variable", cat_cols_all, index=0)
    with cx2:
        y_col = st.selectbox("Column variable", cat_cols_all, index=13)

    if x_col != y_col:
        cross = pd.crosstab(df[x_col], df[y_col], normalize="index").round(3) * 100
        cross.index = cross.index.astype(str).str.replace("_"," ")
        cross.columns = cross.columns.astype(str).str.replace("_"," ")
        fig_ct = px.imshow(cross, text_auto=".0f", aspect="auto",
                           color_continuous_scale="Blues",
                           title=f"{x_col.replace('_',' ').title()} × {y_col.replace('_',' ').title()} (row %)",
                           labels=dict(color="%"))
        fig_ct.update_layout(height=380, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_ct, use_container_width=True)
    else:
        st.warning("Select two different variables.")

    # ── Brand openness → interest ─────────────────────────────────────────────
    st.markdown("<div class='sec'>Brand Openness → Zaria Interest Rate</div>", unsafe_allow_html=True)
    bo_order = ["Very_Loyal_One_Brand","Loyal_But_Open","Multi_Brand_Shopper","No_Brand_Preference"]
    bo_int   = df[df["zaria_interest_label"]=="Interested"]["brand_openness"].value_counts()
    bo_tot   = df["brand_openness"].value_counts()
    bo_df = pd.DataFrame({
        "Segment":  [b.replace("_"," ") for b in bo_order],
        "Interest%": [(bo_int.get(b,0)/max(bo_tot.get(b,1),1)*100) for b in bo_order],
    })
    fig_bo = px.bar(bo_df, x="Segment", y="Interest%",
                    color="Interest%", color_continuous_scale=["#FCEBEB","#1D9E75"],
                    text="Interest%", title="% Interested by Brand Openness")
    fig_bo.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bo.update_layout(height=340, coloraxis_showscale=False,
                         margin=dict(t=50,b=10,l=10,r=10), xaxis_title="", yaxis_title="%")
    st.plotly_chart(fig_bo, use_container_width=True)

    # ── Fashion identity → interest ───────────────────────────────────────────
    st.markdown("<div class='sec'>Fashion Identity × Interest Rate</div>", unsafe_allow_html=True)
    fi_int = (df.groupby("fashion_identity")["zaria_interest_label"]
               .apply(lambda x: (x=="Interested").sum()/len(x)*100)
               .round(1).reset_index())
    fi_int.columns = ["Fashion Identity","Interest Rate %"]
    fi_int["Fashion Identity"] = fi_int["Fashion Identity"].str.replace("_"," ")
    fi_int = fi_int.sort_values("Interest Rate %")
    fig_fi = px.bar(fi_int, x="Interest Rate %", y="Fashion Identity", orientation="h",
                    color="Interest Rate %", color_continuous_scale=["#FAEEDA","#412402"],
                    text="Interest Rate %", title="Interest Rate by Fashion Identity")
    fig_fi.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_fi.update_layout(height=340, coloraxis_showscale=False,
                         margin=dict(t=50,b=10,l=10,r=10), yaxis_title="")
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Income vs Spend boxplot ───────────────────────────────────────────────
    st.markdown("<div class='sec'>Income Band vs Annual Spend</div>", unsafe_allow_html=True)
    inc_order = ["Below_20K","20K_40K","40K_70K","70K_120K","Above_120K"]
    df_sp = df.dropna(subset=["monthly_income_band"]).copy()
    df_sp = df_sp[df_sp["monthly_income_band"].isin(inc_order)].copy()
    df_sp["monthly_income_band"] = pd.Categorical(
        df_sp["monthly_income_band"], categories=inc_order, ordered=True)
    fig_sp = px.box(df_sp, x="monthly_income_band", y="estimated_annual_spend",
                    color="zaria_interest_label",
                    color_discrete_map={"Interested":"#1D9E75",
                                        "Neutral":"#FFC107",
                                        "Not_Interested":"#E24B4A"},
                    title="Annual Spend Distribution by Income & Interest",
                    labels={"monthly_income_band":"Income Band",
                            "estimated_annual_spend":"Est. Annual Spend (₹)"})
    fig_sp.update_layout(height=400, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_sp, use_container_width=True)

    # ── Conversion trigger effectiveness ──────────────────────────────────────
    st.markdown("<div class='sec'>Conversion Trigger Effectiveness</div>", unsafe_allow_html=True)
    ct_eff = (df.groupby("conversion_trigger")["zaria_interest_label"]
               .apply(lambda x: (x=="Interested").sum()/len(x)*100)
               .round(1).sort_values(ascending=False).reset_index())
    ct_eff.columns = ["Trigger","Interest %"]
    ct_eff["Trigger"] = ct_eff["Trigger"].str.replace("_"," ")
    fig_cte = px.bar(ct_eff, x="Interest %", y="Trigger", orientation="h",
                     color="Interest %", color_continuous_scale=["#E6F1FB","#042C53"],
                     text="Interest %", title="% Interested by Conversion Trigger")
    fig_cte.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_cte.update_layout(height=380, coloraxis_showscale=False,
                          margin=dict(t=50,b=10,l=10,r=10), yaxis_title="")
    st.plotly_chart(fig_cte, use_container_width=True)
    st.markdown("<div class='ibox'>💡 Triggers with highest interest rate = first-message content for paid acquisition campaigns.</div>", unsafe_allow_html=True)
