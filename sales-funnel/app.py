"""
Sales Funnel Conversion Analytics — Streamlit demo

Team consulting-style analysis of a fictional multi-dealership auto retail
network's sales funnel (Prospect -> Opportunity -> Sale). Combines
statistical/ML modeling (MCA, Decision Tree, Random Forest, LDA with SMOTE)
with business-oriented insights (cost per channel, revenue-at-risk,
lead scoring).

Note: all data is synthetic — see data/generate_data.py.
Run with: streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import prince
import streamlit as st

from modeling import run_all_models

st.set_page_config(page_title="Sales Funnel Conversion Analytics", layout="wide")

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_data():
    if not (DATA_DIR / "prospects.csv").exists():
        subprocess.run([sys.executable, str(DATA_DIR / "generate_data.py")], check=True)
    prospects = pd.read_csv(DATA_DIR / "prospects.csv")
    opportunities = pd.read_csv(DATA_DIR / "opportunities.csv")
    return prospects, opportunities


prospects, opportunities = load_data()

st.title("🚗 Sales Funnel Conversion Analytics")
st.caption(
    "Demo dashboard — synthetic data, inspired by a real statistical-consulting project "
    "(sales funnel conversion modeling for a multi-dealership retail network). "
    "Combines statistical/ML modeling with business-oriented insights."
)

tab_overview, tab_multivariate, tab_models, tab_business = st.tabs(
    ["📊 Funnel Overview", "🔎 Multivariate (MCA)", "🤖 Predictive Models", "💰 Business Insights"]
)

# ====================================================================
# TAB 1 — FUNNEL OVERVIEW
# ====================================================================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prospects", f"{len(prospects):,}")
    c2.metric("Opportunities", f"{len(opportunities):,}")
    c3.metric("Prospect → Opportunity", f"{prospects['converted'].mean():.1%}")
    c4.metric("Opportunity → Sale", f"{opportunities['sale'].mean():.1%}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Conversion rate by campaign")
        conv_by_campaign = (
            prospects.groupby("campaign")["converted"].mean().sort_values(ascending=False).reset_index()
        )
        fig = px.bar(conv_by_campaign, x="campaign", y="converted",
                     labels={"converted": "Conversion rate", "campaign": "Campaign"})
        fig.update_layout(xaxis_tickangle=-30, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Conversion rate by region")
        conv_by_region = (
            prospects.groupby("region")["converted"].mean().sort_values(ascending=False).reset_index()
        )
        fig2 = px.bar(conv_by_region, x="region", y="converted",
                      labels={"converted": "Conversion rate", "region": "Region"},
                      color_discrete_sequence=["#2E7D32"])
        fig2.update_layout(xaxis_tickangle=-30, yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Prospect volume by brand interest")
        brand_counts = prospects["brand_interest"].value_counts(normalize=True).reset_index()
        brand_counts.columns = ["brand_interest", "share"]
        fig3 = px.bar(brand_counts, x="brand_interest", y="share",
                      labels={"share": "Share of prospects", "brand_interest": "Brand"})
        fig3.update_layout(xaxis_tickangle=-30, yaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Sale rate by owner role")
        sale_by_role = opportunities.groupby("owner_role")["sale"].mean().sort_values(ascending=False).reset_index()
        fig4 = px.bar(sale_by_role, x="owner_role", y="sale",
                      labels={"sale": "Sale rate", "owner_role": "Owner role"},
                      color_discrete_sequence=["#6A4C93"])
        fig4.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Conversion trend over time")
    prospects["created_date"] = pd.to_datetime(prospects["created_date"])
    trend = (
        prospects.set_index("created_date")
        .resample("W")["converted"]
        .mean()
        .reset_index()
    )
    fig5 = px.line(trend, x="created_date", y="converted",
                   labels={"converted": "Conversion rate", "created_date": "Week"})
    fig5.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig5, use_container_width=True)

# ====================================================================
# TAB 2 — MULTIVARIATE ANALYSIS (MCA)
# ====================================================================
with tab_multivariate:
    st.subheader("Multiple Correspondence Analysis (MCA)")
    st.caption(
        "MCA reveals which categories of different variables tend to co-occur — "
        "categories plotted close together share a similar profile."
    )

    stage_choice = st.radio("Funnel stage", ["Prospect", "Opportunity"], horizontal=True)
    source_df = prospects if stage_choice == "Prospect" else opportunities

    cat_cols = ["brand_interest", "region", "campaign", "registration_type", "owner_role"]
    mca = prince.MCA(n_components=2, random_state=42)
    mca = mca.fit(source_df[cat_cols].astype(str))
    coords = mca.column_coordinates(source_df[cat_cols].astype(str)).reset_index()
    coords.columns = ["category", "dim1", "dim2"]
    coords["variable"] = coords["category"].str.split("__").str[0]
    coords["label"] = coords["category"].str.split("__").str[1]

    eig = mca.eigenvalues_summary
    var1 = eig.iloc[0, 1] if len(eig) > 0 else "-"
    var2 = eig.iloc[1, 1] if len(eig) > 1 else "-"

    fig_mca = px.scatter(
        coords, x="dim1", y="dim2", color="variable", text="label",
        labels={"dim1": f"Dimension 1 ({var1})", "dim2": f"Dimension 2 ({var2})"},
        title=f"MCA — {stage_choice} stage",
    )
    fig_mca.update_traces(textposition="top center")
    fig_mca.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_mca.add_vline(x=0, line_dash="dot", line_color="gray")
    fig_mca.update_layout(height=650)
    st.plotly_chart(fig_mca, use_container_width=True)

    st.info(
        "Reading the plot: categories clustered together (e.g. a brand, a region, and a "
        "campaign appearing near each other) tend to co-occur in the same records. "
        "Categories on opposite sides of the origin are negatively associated."
    )

# ====================================================================
# TAB 3 — PREDICTIVE MODELS
# ====================================================================
with tab_models:
    st.subheader("Predictive classification models")
    st.caption(
        "Decision Tree, Random Forest, and LDA, trained with SMOTE oversampling to "
        "address class imbalance — mirroring the methodology of the original consulting project."
    )

    model_stage = st.radio("Predict", ["Prospect → Opportunity", "Opportunity → Sale"], horizontal=True, key="model_stage")
    use_smote = st.checkbox("Use SMOTE (class balancing)", value=True)

    if model_stage == "Prospect → Opportunity":
        model_df = prospects
        feature_cols = ["brand_interest", "region", "campaign", "registration_type",
                         "owner_role", "days_in_prospect_stage"]
        target_col = "converted"
    else:
        model_df = opportunities
        feature_cols = ["brand_interest", "region", "campaign", "registration_type",
                         "owner_role", "days_in_opportunity_stage", "kept_same_brand"]
        target_col = "sale"

    with st.spinner("Training models..."):
        results = run_all_models(model_df, feature_cols, target_col, use_smote=use_smote)

    metrics_df = pd.DataFrame([
        {
            "Model": r.name, "Accuracy": r.accuracy, "Sensitivity": r.sensitivity,
            "Specificity": r.specificity, "PPV": r.ppv, "NPV": r.npv, "ROC AUC": r.roc_auc,
        }
        for r in results.values()
    ]).set_index("Model")
    st.dataframe(metrics_df.style.format("{:.1%}"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC curves")
        fig_roc = go.Figure()
        for r in results.values():
            fig_roc.add_trace(go.Scatter(x=r.fpr, y=r.tpr, name=f"{r.name} (AUC={r.roc_auc:.2f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Random"))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=450)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        best_model_name = metrics_df["ROC AUC"].idxmax()
        st.subheader(f"Confusion matrix — {best_model_name} (best AUC)")
        conf = results[best_model_name].confusion
        fig_cm = px.imshow(
            conf, text_auto=True, color_continuous_scale="Greens",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No", "Yes"], y=["No", "Yes"],
        )
        fig_cm.update_layout(height=450)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Feature importance (Random Forest)")
    rf_importance = results["Random Forest"].feature_importance
    if rf_importance is not None:
        top_n = rf_importance.head(12).reset_index()
        top_n.columns = ["feature", "importance"]
        fig_imp = px.bar(top_n, x="importance", y="feature", orientation="h")
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=450)
        st.plotly_chart(fig_imp, use_container_width=True)

# ====================================================================
# TAB 4 — BUSINESS INSIGHTS
# ====================================================================
with tab_business:
    st.subheader("From statistics to business decisions")
    st.caption(
        "⚠️ Cost and revenue figures below are **illustrative assumptions** (set in the sidebar), "
        "not real financial data — the point is to show how funnel statistics translate into "
        "the kind of economic reasoning a business stakeholder needs."
    )

    st.sidebar.header("Business assumptions")
    avg_margin = st.sidebar.number_input("Avg. gross margin per sale (USD)", value=1800, step=100)
    cost_per_lead = {
        "Dealer Website": 8, "Brand Website": 10, "Google - Mixed": 15, "Google - Commercial": 22,
        "Facebook - Mixed": 9, "Facebook - Commercial": 14, "Referral": 3, "Walk-in": 0, "WhatsApp": 5,
    }
    st.sidebar.caption("Assumed avg. cost per lead by channel (USD) — editable illustrative values")
    for k in cost_per_lead:
        cost_per_lead[k] = st.sidebar.number_input(f"  {k}", value=cost_per_lead[k], step=1, key=f"cost_{k}")

    # --- Cost per acquisition (CPA) and ROI by campaign -----------------
    campaign_stats = prospects.groupby("campaign").agg(
        n_leads=("prospect_id", "count"),
        conv_rate=("converted", "mean"),
    ).reset_index()
    campaign_stats["cost_per_lead"] = campaign_stats["campaign"].map(cost_per_lead)
    campaign_stats["total_spend"] = campaign_stats["n_leads"] * campaign_stats["cost_per_lead"]
    campaign_stats["n_opportunities"] = (campaign_stats["n_leads"] * campaign_stats["conv_rate"]).round()
    # Approximate opportunity->sale rate from the overall opportunities dataset
    overall_sale_rate = opportunities["sale"].mean()
    campaign_stats["est_sales"] = (campaign_stats["n_opportunities"] * overall_sale_rate).round()
    campaign_stats["cpa"] = campaign_stats["total_spend"] / campaign_stats["est_sales"].replace(0, np.nan)
    campaign_stats["est_revenue"] = campaign_stats["est_sales"] * avg_margin
    campaign_stats["roi"] = (campaign_stats["est_revenue"] - campaign_stats["total_spend"]) / campaign_stats["total_spend"].replace(0, np.nan)

    st.subheader("💵 Estimated cost per acquisition (CPA) and ROI by channel")
    display_cols = ["campaign", "n_leads", "conv_rate", "total_spend", "est_sales", "cpa", "roi"]
    st.dataframe(
        campaign_stats[display_cols].sort_values("roi", ascending=False).style.format({
            "conv_rate": "{:.1%}", "total_spend": "${:,.0f}", "cpa": "${:,.0f}",
            "roi": "{:.0%}", "est_sales": "{:,.0f}",
        }),
        use_container_width=True,
    )

    fig_roi = px.bar(
        campaign_stats.sort_values("roi", ascending=False),
        x="campaign", y="roi", labels={"roi": "Estimated ROI", "campaign": "Channel"},
        color="roi", color_continuous_scale="RdYlGn",
    )
    fig_roi.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30)
    st.plotly_chart(fig_roi, use_container_width=True)

    st.divider()

    # --- Revenue at risk from funnel drop-off ---------------------------
    st.subheader("📉 Revenue at risk from funnel drop-off")
    n_prospects = len(prospects)
    n_converted = prospects["converted"].sum()
    n_opportunities = len(opportunities)
    n_sales = opportunities["sale"].sum()

    lost_at_prospect = n_prospects - n_converted
    lost_at_opportunity = n_opportunities - n_sales

    c1, c2, c3 = st.columns(3)
    c1.metric("Lost at Prospect stage", f"{lost_at_prospect:,}")
    c2.metric("Lost at Opportunity stage", f"{lost_at_opportunity:,}")
    c3.metric("Est. revenue at risk (Opportunity stage)", f"${lost_at_opportunity * avg_margin:,.0f}")

    st.caption(
        f"If just a **5-point improvement** in Opportunity→Sale conversion were achieved "
        f"(from {overall_sale_rate:.1%} to {overall_sale_rate + 0.05:.1%}), holding lead volume constant, "
        f"that implies roughly **{int(n_opportunities * 0.05):,} additional sales** — "
        f"about **${int(n_opportunities * 0.05 * avg_margin):,}** in additional estimated margin."
    )

    st.divider()

    # --- Lead scoring / prioritization -----------------------------------
    st.subheader("🎯 Lead prioritization (predicted conversion probability)")
    st.caption(
        "Using the Random Forest model's predicted probabilities to rank currently-open "
        "prospects — the kind of output a sales team could use to prioritize outreach."
    )

    prospect_features = ["brand_interest", "region", "campaign", "registration_type", "owner_role", "days_in_prospect_stage"]
    X_all = pd.get_dummies(prospects[prospect_features], drop_first=True)
    y_all = prospects["converted"]

    # Re-fit a fresh RF on the full dataset purely for scoring open leads
    from sklearn.ensemble import RandomForestClassifier
    scorer = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    scorer.fit(X_all, y_all)

    open_leads = prospects[prospects["converted"] == 0].copy()
    open_leads_X = pd.get_dummies(open_leads[prospect_features], drop_first=True)
    open_leads_X = open_leads_X.reindex(columns=X_all.columns, fill_value=0)
    open_leads["predicted_conversion_prob"] = scorer.predict_proba(open_leads_X)[:, 1]

    top_leads = open_leads.sort_values("predicted_conversion_prob", ascending=False).head(10)
    st.dataframe(
        top_leads[["prospect_id", "dealership", "brand_interest", "region", "campaign",
                    "days_in_prospect_stage", "predicted_conversion_prob"]]
        .style.format({"predicted_conversion_prob": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption("Top 10 currently-open prospects ranked by predicted probability of converting.")




