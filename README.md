# Student Engagement Analysis

> Predicting student participation in learning opportunities using machine learning on real sign-up and engagement data.

---

## Project Overview

This project analyses historical student sign-up and participation records to:

- Understand engagement patterns across opportunity types, countries, and demographics
- Identify the key drivers of student participation
- Build a predictive model that estimates participation likelihood for future opportunities
- Deliver actionable recommendations for program managers

**Best model achieved: AUC = 0.917 (Gradient Boosting)**

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_G-Baloyi/Student_Engagement.git
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

## Dataset

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

## Models Trained

| Model | Test AUC | CV AUC (5-fold) |
|---|---|---|
| Logistic Regression | 0.878 | ~0.87 |
| Random Forest | 0.912 | ~0.91 |
| **Gradient Boosting** | **0.917** | **~0.91** |

**Target variable:** `is_participated` — 1 if status is Team Allocated, Started, or Rewards Award

---

## Key Findings

1. **41.7% rejection rate** — the single largest status group; pre-screening could dramatically improve fit
2. **Events & Competitions** outperform Internships in participation rate
3. **Nigeria and Ghana** show above-average participation rates despite lower sign-up volumes
4. **Signup-to-apply lag** is a key behavioural signal — faster applicants are more likely to participate
5. **Opportunity category and country** are the most predictive features

---

## Recommendations

| # | Recommendation | KPI |
|---|---|---|
| R1 | Increase event/competition frequency as gateway experiences | Event → Internship conversion ≥ 20% |
| R2 | Add eligibility pre-screening to reduce rejections | Rejection rate < 25% |
| R3 | Automated nudge emails 24–48 hrs after sign-up | Median days-to-apply ≤ 3 |
| R4 | Targeted outreach in high-converting countries (NG, GH) | Country-level participation tracked quarterly |
| R5 | Use ML model scores to prioritise resource allocation | Model AUC ≥ 0.90 per cohort |

---

## Dependencies

See `requirements.txt`. Core libraries:

- `pandas` · `numpy` · `matplotlib` · `seaborn`
- `scikit-learn` (ML models, metrics, preprocessing)
- `jupyterlab` / `notebook` (interactive notebooks)

---

