import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import load_data, get_feature_matrix

@st.cache_resource
def train_regressors():
    df = load_data()
    X, feat_cols = get_feature_matrix(df)
    X = X.fillna(0)
    y_raw = df["estimated_annual_spend"].values
    y     = np.log1p(y_raw)                          # log-transform skewed target

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    models = {
        "Linear Regression": (LinearRegression(),  True),
        "Ridge Regression":  (Ridge(alpha=1.0),     True),
        "Random Forest Reg": (RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1), False),
    }

    results, trained = {}, {}
    for name, (mdl, use_sc) in models.items():
        Xtr_ = X_tr_sc if use_sc else X_tr
        Xte_ = X_te_sc if use_sc else X_te
        mdl.fit(Xtr_, y_tr)
        yp   = mdl.predict(Xte_)
        yp_r = np.expm1(yp)
        yt_r = np.expm1(y_te)
        results[name] = dict(
            rmse    = float(np.sqrt(mean_squared_error(yt_r, yp_r))),
            mae     = float(mean_absolute_error(yt_r, yp_r)),
            r2      = float(r2_score(y_te, yp)),
            y_pred  = yp_r,
            y_actual= yt_r,
        )
        trained[name] = (mdl, sc if use_sc else None)

    # Save RF for predictor
    rf_mdl, _ = trained["Random Forest Reg"]
    joblib.dump({"model": rf_mdl, "feat_cols": feat_cols,
                 "log": True}, "zaria_reg.pkl")

    return trained, results, feat_cols, X, df

def render():
    st.title("💰 Regression & Customer Lifetime Value")
    st.caption("How much will a customer spend annually? — Linear · Ridge · Random Forest Regressor")

    with st.spinner("Training regression models…"):
        trained, results, feat_cols, X_full, df_full = train_regressors()

    # ── Model comparison ──────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Model Performance Comparison</div>", unsafe_allow_html=True)
    cmp_df = pd.DataFrame({
        "Model":   list(results.keys()),
        "RMSE ₹":  [results[m]["rmse"] for m in results],
        "MAE ₹":   [results[m]["mae"]  for m in results],
        "R²":      [results[m]["r2"]   for m in results],
    })
    cmp_df["RMSE ₹"] = cmp_df["RMSE ₹"].map("₹{:,.0f}".format)
    cmp_df["MAE ₹"]  = cmp_df["MAE ₹"].map("₹{:,.0f}".format)
    cmp_df["R²"]     = cmp_df["R²"].map("{:.4f}".format)
    st.dataframe(cmp_df, use_container_width=True)

    r2_df = pd.DataFrame({"Model": list(results.keys()),
                           "R² Score": [results[m]["r2"] for m in results]})
    fig_r2 = px.bar(r2_df, x="Model", y="R² Score", color="Model",
                    color_discrete_sequence=["#1D9E75","#378ADD","#BA7517"],
                    text_auto=".3f", title="R² Score (higher = better fit)")
    fig_r2.update_traces(textposition="outside")
    fig_r2.update_layout(height=300, showlegend=False, yaxis_range=[0,1],
                         margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_r2, use_container_width=True)

    # ── Model deep-dive ───────────────────────────────────────────────────────
    sel = st.selectbox("Select model for detailed analysis", list(results.keys()))
    res = results[sel]
    m1,m2,m3 = st.columns(3)
    m1.metric("RMSE", f"₹{res['rmse']:,.0f}")
    m2.metric("MAE",  f"₹{res['mae']:,.0f}")
    m3.metric("R²",   f"{res['r2']:.4f}")

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    st.markdown("<div class='sec'>Actual vs Predicted Annual Spend</div>", unsafe_allow_html=True)
    ap = pd.DataFrame({"Actual ₹": res["y_actual"], "Predicted ₹": res["y_pred"]})
    cap = ap["Actual ₹"].quantile(0.97)
    ap  = ap[ap["Actual ₹"] <= cap]
    fig_ap = px.scatter(ap, x="Actual ₹", y="Predicted ₹", opacity=0.4,
                        color_discrete_sequence=["#1D9E75"],
                        title=f"Actual vs Predicted — {sel}")
    mx = max(ap["Actual ₹"].max(), ap["Predicted ₹"].max())
    fig_ap.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode="lines",
                                line=dict(dash="dash",color="red",width=1.5),
                                name="Perfect fit"))
    fig_ap.update_layout(height=420, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_ap, use_container_width=True)

    # ── Residuals ─────────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Residual Analysis</div>", unsafe_allow_html=True)
    ap["Residual ₹"] = ap["Predicted ₹"] - ap["Actual ₹"]
    r1, r2 = st.columns(2)
    with r1:
        fig_rs = px.scatter(ap, x="Predicted ₹", y="Residual ₹", opacity=0.4,
                            color_discrete_sequence=["#378ADD"],
                            title="Residuals vs Predicted")
        fig_rs.add_hline(y=0, line_dash="dash", line_color="red")
        fig_rs.update_layout(height=320, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_rs, use_container_width=True)
    with r2:
        fig_rh = px.histogram(ap, x="Residual ₹", nbins=30,
                              color_discrete_sequence=["#BA7517"],
                              title="Residual Distribution")
        fig_rh.update_layout(height=320, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_rh, use_container_width=True)

    # ── Feature importance (RF only) ──────────────────────────────────────────
    st.markdown("<div class='sec'>Feature Importance — Random Forest Regressor</div>", unsafe_allow_html=True)
    rf_mdl, _ = trained["Random Forest Reg"]
    clean = [c.replace("_enc","").replace("_"," ").title() for c in feat_cols]
    fi_df = (pd.DataFrame({"Feature": clean,
                            "Importance": rf_mdl.feature_importances_})
               .sort_values("Importance", ascending=True).tail(15))
    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale=["#FAC775","#412402"],
                    text="Importance", title="Top-15 Spend Predictors")
    fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_fi.update_layout(height=460, coloraxis_showscale=False,
                         margin=dict(t=50,b=10,l=10,r=10), yaxis_title="")
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── CLV tiering ───────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Customer Lifetime Value — Tiering & Playbook</div>", unsafe_allow_html=True)
    df_full = df_full.copy()
    X_all = get_feature_matrix(df_full)[0].fillna(0)
    df_full["pred_spend"] = np.expm1(rf_mdl.predict(X_all)).round(-1)

    p80 = df_full["pred_spend"].quantile(0.80)
    p30 = df_full["pred_spend"].quantile(0.30)
    df_full["clv_tier"] = pd.cut(df_full["pred_spend"],
                                  bins=[0, p30, p80, 1e9],
                                  labels=["Standard","Mid-Value","VIP"])

    t1, t2 = st.columns(2)
    with t1:
        tier_cnt = df_full["clv_tier"].value_counts().reset_index()
        tier_cnt.columns = ["Tier","Count"]
        fig_tc = px.pie(tier_cnt, names="Tier", values="Count",
                        color_discrete_sequence=["#1D9E75","#378ADD","#888780"],
                        title="CLV Tier Distribution", hole=0.4)
        fig_tc.update_layout(height=300, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_tc, use_container_width=True)

    with t2:
        tier_sp = (df_full.groupby("clv_tier")["pred_spend"]
                    .mean().round(0).reset_index())
        tier_sp.columns = ["Tier","Avg Spend ₹"]
        fig_ts = px.bar(tier_sp, x="Tier", y="Avg Spend ₹", color="Tier",
                        color_discrete_sequence=["#1D9E75","#378ADD","#888780"],
                        text="Avg Spend ₹", title="Avg Predicted Spend by Tier")
        fig_ts.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig_ts.update_layout(height=300, showlegend=False,
                             margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_ts, use_container_width=True)

    # Playbook cards
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown(f"<div class='ibox'><b>🏆 VIP (top 20%)</b><br>Spend > ₹{p80:,.0f}<br>Count: {(df_full['clv_tier']=='VIP').sum()}<br><br>✅ Free express delivery<br>✅ Personal stylist WhatsApp<br>✅ Early collection preview<br>✅ Handwritten thank-you</div>", unsafe_allow_html=True)
    with v2:
        st.markdown(f"<div class='wbox'><b>🥈 Mid-Value (50%)</b><br>₹{p30:,.0f}–₹{p80:,.0f}<br>Count: {(df_full['clv_tier']=='Mid-Value').sum()}<br><br>✅ Email campaigns<br>✅ Instagram content<br>✅ Seasonal discounts<br>✅ Loyalty points</div>", unsafe_allow_html=True)
    with v3:
        st.markdown(f"<div class='ibox' style='background:#f8f9fa;border-color:#dee2e6'><b>📢 Standard (30%)</b><br>Spend < ₹{p30:,.0f}<br>Count: {(df_full['clv_tier']=='Standard').sum()}<br><br>✅ WhatsApp broadcast<br>✅ Festival announcements<br>✅ No paid retargeting<br>✅ Review after 6 months</div>", unsafe_allow_html=True)

    with st.expander("🏆 Top 20 Highest CLV Customers"):
        show_cols = ["age_group","region","city_tier","fashion_identity",
                     "brand_openness","monthly_income_band","pred_spend","clv_tier"]
        show_cols = [c for c in show_cols if c in df_full.columns]
        st.dataframe(
            df_full.sort_values("pred_spend",ascending=False)[show_cols].head(20).reset_index(drop=True),
            use_container_width=True,
        )
    st.download_button("Download CLV Rankings",
                       df_full[show_cols].to_csv(index=False).encode(),
                       "zaria_clv.csv","text/csv")
