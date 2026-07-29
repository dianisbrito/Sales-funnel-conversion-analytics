# Sales Funnel Conversion Analytics

**Author:** Diana Brito Hoyos — Biologist & Biostatistician | Data Analyst

🔗 **[Live demo](https://sales-funnel-conversion-analytics.streamlit.app/)**

An interactive analytics dashboard for a Prospect → Opportunity → Sale conversion funnel, combining classical statistical/ML methods with business-oriented economic insights (cost per acquisition, ROI by channel, revenue at risk, lead scoring).

> ⚠️ **Note on data and origin:** This project is inspired by a real applied-statistics **team consulting project** (Statistical Consulting course, Master's in Applied Statistics, UNC), originally conducted for a multi-dealership auto retail client by an 8-person consulting team including myself. **All data, the client's name, and dealership names in this repository are entirely synthetic** (see [`data/generate_data.py`](.sales-funnel/data/generate_data.py)) — no real business, financial, or client data is used. The company here is referred to generically as "AutoCo". The methodology (funnel analysis, MCA, SMOTE-balanced classification, model comparison) mirrors the original project's approach; the business-insights layer (CPA, ROI, revenue-at-risk, lead scoring) was added for this portfolio version to demonstrate a more applied, business-facing extension of that analysis.

---

## What this demonstrates

- **Statistical rigor**: Multiple Correspondence Analysis (MCA) for exploring categorical associations; classification models evaluated with accuracy, sensitivity, specificity, PPV/NPV, and ROC AUC — not just a single "accuracy" number.
- **Applied machine learning**: Decision Tree, Random Forest, and Linear Discriminant Analysis, each trained with **SMOTE** oversampling to address the class imbalance typical of conversion funnels (most prospects don't convert).
- **Business translation**: statistics alone don't drive decisions — the Business Insights tab translates model output and funnel metrics into cost-per-acquisition, ROI by marketing channel, revenue-at-risk from funnel drop-off, and a lead-prioritization score.

## Dashboard tabs

1. **📊 Funnel Overview** — conversion rates by campaign, region, brand, and owner role; conversion trend over time.
2. **🔎 Multivariate (MCA)** — interactive Multiple Correspondence Analysis, switchable between the Prospect and Opportunity stages.
3. **🤖 Predictive Models** — Decision Tree / Random Forest / LDA comparison (with optional SMOTE), ROC curves, confusion matrix, and feature importance.
4. **💰 Business Insights** — editable cost-per-lead assumptions (sidebar), estimated CPA and ROI by channel, revenue-at-risk from funnel drop-off, and a Random Forest–based lead-prioritization table.

## Project structure

```
sales-funnel/
├── README.md
├── requirements.txt
├── app.py                     ← Streamlit dashboard (4 tabs)
├── modeling.py                ← reusable modeling module (SMOTE + 3 classifiers + metrics)
└── data/
    ├── generate_data.py       ← synthetic data generator
    ├── prospects.csv          ← generated
    └── opportunities.csv      ← generated
```

## Running locally

```bash
pip install -r requirements.txt
python data/generate_data.py     # generates data/prospects.csv and opportunities.csv
streamlit run app.py
```

## Tech stack

`Python` · `scikit-learn` · `imbalanced-learn (SMOTE)` · `prince (MCA)` · `Streamlit` · `pandas` · `plotly`

## About the author

See full profile on [GitHub](https://github.com/dianisbrito/dianisbrito) 
