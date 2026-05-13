# 🎓 RIT Student Engagement Analysis

> Predicting student participation in learning opportunities using machine learning on real RIT sign-up and engagement data.

---

## 📌 Project Overview

This project analyses historical student sign-up and participation records from Rochester Institute of Technology (RIT) to:

- Understand engagement patterns across opportunity types, countries, and demographics
- Identify the key drivers of student participation
- Build a predictive model that estimates participation likelihood for future opportunities
- Deliver actionable recommendations for program managers

**Best model achieved: AUC = 0.917 (Gradient Boosting)**

---

## 📁 Project Structure

```
RIT_Student_Engagement_Project/
│
├── data/
│   ├── raw/                          # Original dataset (do not modify)
│   │   └── RIT_Opportunity_Wise_Data_Sheet.csv
│   └── processed/                    # Cleaned & engineered data
│       ├── RIT_Cleaned_Engineered.csv
│       ├── predictions.csv
│       └── analysis_summary.json
│
├── notebooks/
│   ├── 01_data_understanding.py      # Week 1 – Data exploration & cleaning
│   ├── 02_eda_cohort_analysis.py     # Week 2 – EDA & cohort analysis
│   ├── 03_ml_modeling.py             # Week 3 – ML training & evaluation
│   └── 04_final_insights_recommendations.py  # Week 4 – Insights & recommendations
│
├── outputs/
│   └── figures/                      # All generated charts and visualisations
│       ├── 01_status_distribution.png
│       ├── 02_opportunity_categories.png
│       ├── 03_top_countries.png
│       ├── 04_gender_distribution.png
│       ├── 05_missing_values.png
│       ├── 06_signups_over_time.png
│       ├── 07_participation_by_category.png
│       ├── 08_participation_by_country.png
│       ├── 09_age_distribution.png
│       ├── 10_top_opportunities.png
│       ├── 11_heatmap_category_country.png
│       ├── 12_signup_to_apply_lag.png
│       ├── 13_model_comparison_auc.png
│       ├── 14_roc_curves.png
│       ├── 15_confusion_matrix.png
│       ├── 16_feature_importance.png
│       ├── 17_predicted_probabilities.png
│       ├── 18_engagement_funnel.png
│       ├── 19_participation_gender_category.png
│       └── 20_participation_trend_year.png
│
├── scripts/
│   ├── data_preprocessing.py         # Standalone cleaning pipeline
│   ├── train_model.py                # Standalone model training
│   └── generate_figures.py           # Re-generate all figures
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/RIT_Student_Engagement_Project.git
cd RIT_Student_Engagement_Project
pip install -r requirements.txt
```

### 2. Run the pipeline (scripts)

```bash
# Step 1 – Clean & engineer features
python scripts/data_preprocessing.py

# Step 2 – Train models & save predictions
python scripts/train_model.py

# Step 3 – Regenerate all figures
python scripts/generate_figures.py
```

### 3. Run notebooks interactively

The `.py` notebook files are written in the [Percent Format](https://jupytext.readthedocs.io/) and can be opened directly as Jupyter notebooks:

```bash
# Open any notebook
jupyter notebook notebooks/01_data_understanding.py
```

Or convert to `.ipynb` with Jupytext:

```bash
pip install jupytext
jupytext --to notebook notebooks/01_data_understanding.py
```

---

## 📊 Dataset

| Field | Description |
|---|---|
| `Learner SignUp DateTime` | When the student joined the platform |
| `Opportunity Name` | Name of the learning opportunity |
| `Opportunity Category` | Course / Internship / Event / Competition / Engagement |
| `Status Description` | Participation status (Rejected, Team Allocated, Started, etc.) |
| `Country` | Student's country |
| `Gender` | Student-reported gender |
| `Date of Birth` | Used to derive age |
| `Apply Date` | When the student applied |
| `Institution Name` | Student's academic institution |

**Size:** 8,558 records across 16 features | **Opportunities:** 69 unique programs

---

## 🤖 Models Trained

| Model | Test AUC | CV AUC (5-fold) |
|---|---|---|
| Logistic Regression | 0.878 | ~0.87 |
| Random Forest | 0.912 | ~0.91 |
| **Gradient Boosting** ✅ | **0.917** | **~0.91** |

**Target variable:** `is_participated` — 1 if status is Team Allocated, Started, or Rewards Award

---

## 💡 Key Findings

1. **41.7% rejection rate** — the single largest status group; pre-screening could dramatically improve fit
2. **Events & Competitions** outperform Internships in participation rate
3. **Nigeria and Ghana** show above-average participation rates despite lower sign-up volumes
4. **Signup-to-apply lag** is a key behavioural signal — faster applicants are more likely to participate
5. **Opportunity category and country** are the most predictive features

---

## 📋 Recommendations

| # | Recommendation | KPI |
|---|---|---|
| R1 | Increase event/competition frequency as gateway experiences | Event → Internship conversion ≥ 20% |
| R2 | Add eligibility pre-screening to reduce rejections | Rejection rate < 25% |
| R3 | Automated nudge emails 24–48 hrs after sign-up | Median days-to-apply ≤ 3 |
| R4 | Targeted outreach in high-converting countries (NG, GH) | Country-level participation tracked quarterly |
| R5 | Use ML model scores to prioritise resource allocation | Model AUC ≥ 0.90 per cohort |

---

## 📦 Dependencies

See `requirements.txt`. Core libraries:

- `pandas` · `numpy` · `matplotlib` · `seaborn`
- `scikit-learn` (ML models, metrics, preprocessing)
- `jupyterlab` / `notebook` (interactive notebooks)

---

## 👤 Author

**Gee**  
RIT Excelerate Internship – Data Science Track  
Analysis Period: 2022–2024 cohort data

---

## 📄 License

This project is for educational purposes as part of the RIT Excelerate Internship programme.
