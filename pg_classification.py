import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from utils import load_data, get_feature_matrix

# ── Train once, cache ─────────────────────────────────────────────────────────
@st.cache_resource
def train_all():
    df  = load_data()
    X, feat_cols = get_feature_matrix(df)
    X = X.fillna(0)

    le = LabelEncoder()
    y  = le.fit_transform(df["zaria_interest_label"].astype(str))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    sm = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric="mlogloss", verbosity=0),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42),
    }

    results, trained = {}, {}
    for name, mdl in models.items():
        mdl.fit(X_tr_sm, y_tr_sm)
        yp   = mdl.predict(X_te)
        yprb = mdl.predict_proba(X_te)
        results[name] = dict(
            accuracy  = accuracy_score(y_te, yp),
            precision = precision_score(y_te, yp, average="weighted", zero_division=0),
            recall    = recall_score(y_te, yp, average="weighted", zero_division=0),
            f1        = f1_score(y_te, yp, average="weighted", zero_division=0),
            y_pred    = yp,
            y_prob    = yprb,
            cm        = confusion_matrix(y_te, yp),
        )
        trained[name] = mdl

    # Save best model for predictor page
    joblib.dump({"model": trained["Random Forest"],
                 "le": le, "feat_cols": feat_cols}, "zaria_clf.pkl")

    return trained, results, X_te, y_te, le, feat_cols, X, df

def render():
    st.title("🎯 Classification")
    st.caption("Will a customer buy from Zaria? — Accuracy · Precision · Recall · F1 · ROC-AUC · Feature Importance")

    with st.spinner("Training classifiers (SMOTE applied)…"):
        trained, results, X_te, y_te, le, feat_cols, X_full, df_full = train_all()

    classes = le.classes_

    # ── Model comparison table ────────────────────────────────────────────────
    st.markdown("<div class='sec'>Model Performance Comparison</div>", unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        "Model":     list(results.keys()),
        "Accuracy":  [results[m]["accuracy"]  for m in results],
        "Precision": [results[m]["precision"] for m in results],
        "Recall":    [results[m]["recall"]    for m in results],
        "F1-Score":  [results[m]["f1"]        for m in results],
    }).round(4)

    def hl(s):
        return ["background-color:#d4edda;font-weight:bold"
                if v == s.max() else "" for v in s]

    st.dataframe(
        metrics_df.style.apply(hl, subset=["Accuracy","Precision","Recall","F1-Score"]),
        use_container_width=True,
    )

    # Grouped bar chart
    ml = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_cmp = px.bar(ml, x="Metric", y="Score", color="Model", barmode="group",
                     text_auto=".3f",
                     color_discrete_sequence=["#1D9E75","#378ADD","#BA7517"],
                     title="Accuracy · Precision · Recall · F1-Score")
    fig_cmp.update_traces(textposition="outside")
    fig_cmp.update_layout(height=380, yaxis_range=[0,1.1],
                          margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Model selector ────────────────────────────────────────────────────────
    st.markdown("---")
    sel = st.selectbox("Select model for detailed analysis", list(results.keys()))
    res = results[sel]
    mdl = trained[sel]

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Accuracy",  f"{res['accuracy']:.4f}")
    m2.metric("Precision", f"{res['precision']:.4f}")
    m3.metric("Recall",    f"{res['recall']:.4f}")
    m4.metric("F1-Score",  f"{res['f1']:.4f}")

    # ── Confusion matrix + ROC ────────────────────────────────────────────────
    st.markdown("<div class='sec'>Confusion Matrix & ROC Curves</div>", unsafe_allow_html=True)
    cm_col, roc_col = st.columns(2)

    with cm_col:
        fig_cm, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Greens",
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, linewidths=0.5, annot_kws={"size":13})
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(f"Confusion Matrix — {sel}", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close()

    with roc_col:
        y_bin  = label_binarize(y_te, classes=list(range(len(classes))))
        y_prob = res["y_prob"]
        roc_colors = ["#1D9E75","#378ADD","#E24B4A"]

        fig_roc = go.Figure()
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{cls} (AUC={roc_auc:.3f})",
                line=dict(color=roc_colors[i % len(roc_colors)], width=2),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            name="Random baseline",
        ))
        fig_roc.update_layout(
            title=f"ROC Curves (one-vs-rest) — {sel}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=380, margin=dict(t=50,b=10,l=10,r=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Feature Importance</div>", unsafe_allow_html=True)
    if hasattr(mdl, "feature_importances_"):
        imps = mdl.feature_importances_
    elif hasattr(mdl, "coef_"):
        imps = np.abs(mdl.coef_).mean(axis=0)
    else:
        imps = np.zeros(len(feat_cols))

    clean = [c.replace("_enc","").replace("_"," ").title() for c in feat_cols]
    fi_df = (pd.DataFrame({"Feature": clean, "Importance": imps})
               .sort_values("Importance", ascending=True).tail(20))

    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale=["#9FE1CB","#085041"],
                    text="Importance",
                    title=f"Top-20 Feature Importances — {sel}")
    fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_fi.update_layout(height=540, coloraxis_showscale=False,
                         margin=dict(t=50,b=10,l=10,r=10), yaxis_title="")
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown("<div class='ibox'>🔑 Top features = what drives purchase intent. These are your marketing brief — invest copy, creatives, and targeting around these signals.</div>", unsafe_allow_html=True)

    # ── Full classification report ────────────────────────────────────────────
    with st.expander("📋 Full Classification Report"):
        cr = classification_report(y_te, res["y_pred"],
                                   target_names=classes, output_dict=True)
        st.dataframe(pd.DataFrame(cr).T.round(3), use_container_width=True)

    # ── Hot leads table ───────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Top 50 High-Probability Leads</div>", unsafe_allow_html=True)
    X_all = get_feature_matrix(df_full)[0].fillna(0)
    rf    = trained["Random Forest"]
    probs = rf.predict_proba(X_all)
    int_idx = list(classes).index("Interested") if "Interested" in classes else 0
    df_full = df_full.copy()
    df_full["interest_prob"]     = probs[:, int_idx].round(3)
    df_full["predicted_label"]   = le.inverse_transform(rf.predict(X_all))
    df_full["lead_priority"]     = pd.cut(
        df_full["interest_prob"],
        bins=[0, 0.40, 0.70, 1.01],
        labels=["Low Priority","Nurture","Hot Lead"],
    )
    leads = df_full.sort_values("interest_prob", ascending=False).head(50)
    show  = ["age_group","region","city_tier","fashion_identity","brand_openness",
             "conversion_trigger","monthly_income_band",
             "interest_prob","predicted_label","lead_priority"]
    show  = [c for c in show if c in leads.columns]
    st.dataframe(leads[show].reset_index(drop=True), use_container_width=True)
    st.download_button("Download All Predictions",
                       df_full[show].to_csv(index=False).encode(),
                       "zaria_clf_predictions.csv","text/csv")
