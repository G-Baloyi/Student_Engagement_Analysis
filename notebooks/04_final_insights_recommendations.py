# %% [markdown]
# # Week 4 – Final Insights, Recommendations & Program Framework
# **RIT Student Engagement Analysis**
#
# This notebook synthesises all prior weeks into:
# 1. Summary of key findings
# 2. Engagement funnel analysis
# 3. Actionable recommendations
# 4. Measurement framework for future cohorts
# 5. Reflection
# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

FIGDIR = '../outputs/figures'
CATS  = ['#1B4F72','#2E86AB','#E84855','#F4A261','#2EC4B6','#8E44AD','#E67E22','#27AE60']
PAL   = {'primary':'#1B4F72','secondary':'#2E86AB','accent':'#E84855',
         'highlight':'#F4A261','success':'#2EC4B6','muted':'#95A5A6'}

plt.rcParams.update({'figure.dpi':150,'font.family':'DejaVu Sans',
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':0.3,'axes.titlesize':14,
                     'axes.titleweight':'bold'})

# %% [markdown]
# ## 1. Load Data & Analysis Summary
# %%
df = pd.read_csv('../data/processed/RIT_Cleaned_Engineered.csv')
with open('../data/processed/analysis_summary.json') as f:
    summary = json.load(f)

print("=== PROJECT SUMMARY ===")
print(f"Total records:               {summary['total_records']:,}")
print(f"Unique opportunities:         {summary['total_opportunities']}")
print(f"Overall participation rate:   {summary['overall_participation_rate']:.1%}")
print(f"\nBest ML model:               {summary['best_model']}")
print(f"Best model AUC:              {summary['best_auc']:.4f}")
print(f"\nModel comparison:")
for name, scores in summary['model_results'].items():
    print(f"  {name:25s}: AUC={scores['auc']:.4f}  CV-AUC={scores['cv_auc']:.4f}")

# %% [markdown]
# ## 2. Engagement Funnel
# %%
statuses = ['Applied','Waitlisted','Team Allocated','Started','Rewards Award']
counts   = [df[df['Status Description']==s].shape[0] for s in statuses]

fig, ax = plt.subplots(figsize=(10,5))
bars = ax.barh(statuses, counts, color=CATS[:5])
ax.set_title('Student Engagement Funnel – All Opportunities')
ax.set_xlabel('Number of Students')
for bar, val in zip(bars, counts):
    ax.text(bar.get_width()+15, bar.get_y()+bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/18_engagement_funnel.png', bbox_inches='tight')
plt.show()
print("Funnel conversion rates:")
for s, c in zip(statuses, counts):
    print(f"  {s:20s}: {c:,}  ({c/counts[0]*100:.1f}% of Applied)")

# %% [markdown]
# ## 3. Participation Trend by Year
# %%
yr_part = df.groupby('signup_year')['is_participated'].mean()
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(yr_part.index, yr_part.values*100, marker='o', linewidth=2.5,
        color=PAL['primary'], markersize=10, markerfacecolor=PAL['accent'])
ax.fill_between(yr_part.index, yr_part.values*100, alpha=0.15, color=PAL['secondary'])
for yr, val in yr_part.items():
    ax.annotate(f'{val*100:.1f}%', (yr, val*100+0.5), ha='center', fontsize=10,
                color=PAL['dark'] if 'dark' in PAL else '#0D1B2A')
ax.set_title('Participation Rate Trend by Year')
ax.set_xlabel('Year'); ax.set_ylabel('Participation Rate (%)')
ax.set_xticks(yr_part.index)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/20_participation_trend_year.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. Top Opportunities by Participation Rate
# %%
opp_stats = df.groupby('Opportunity Name').agg(
    signups=('is_participated','count'),
    part_rate=('is_participated','mean')
).query('signups >= 30').sort_values('part_rate', ascending=False).head(10)

print("Top 10 Opportunities by Participation Rate (min 30 sign-ups):")
print((opp_stats['part_rate']*100).round(1).astype(str)+'%')

fig, ax = plt.subplots(figsize=(11,6))
labels = [n[:45]+'…' if len(n)>45 else n for n in opp_stats.index]
ax.barh(labels, opp_stats['part_rate']*100, color=CATS[:len(opp_stats)])
ax.set_title('Top 10 Opportunities by Participation Rate')
ax.set_xlabel('Participation Rate (%)')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/21_top_opp_by_part_rate.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Gender × Category Participation
# %%
g_cat = df.groupby(['Gender','Opportunity Category'])['is_participated'].mean().unstack()
fig, ax = plt.subplots(figsize=(11,5))
g_cat.T.plot(kind='bar', ax=ax, color=CATS[:len(g_cat)])
ax.set_title('Participation Rate by Gender and Opportunity Category')
ax.set_ylabel('Participation Rate')
ax.tick_params(axis='x', rotation=30)
ax.legend(title='Gender', bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f'{FIGDIR}/19_participation_gender_category.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Actionable Recommendations
# %%
recommendations = {
    "R1 – Target High-Signal Categories": {
        "Finding": "Events and Competitions show higher participation rates than Internships.",
        "Action":  "Increase event frequency and promote competitions as gateway experiences.",
        "Metric":  "Track % of event attendees who subsequently apply to internships."
    },
    "R2 – Reduce Rejection Rate": {
        "Finding": f"41.7% of all records are Rejected — the single largest status group.",
        "Action":  "Implement eligibility pre-screening before students apply to reduce mismatches.",
        "Metric":  "Target rejection rate below 25% within 2 cohorts."
    },
    "R3 – Shorten Sign-up to Apply Gap": {
        "Finding": "Many students sign up but take days or weeks before applying.",
        "Action":  "Send automated nudge emails 24–48 hrs after sign-up to prompt application.",
        "Metric":  "Median days-to-apply; target ≤ 3 days."
    },
    "R4 – Leverage Geographic Diversity": {
        "Finding": "Nigeria and Ghana show strong participation rates despite fewer sign-ups.",
        "Action":  "Run targeted outreach campaigns in high-converting countries.",
        "Metric":  "Participation rate by country cohort, tracked quarterly."
    },
    "R5 – Prioritise Opportunities with AUC > 0.91": {
        "Finding": f"Best model ({summary['best_model']}) achieves AUC={summary['best_auc']:.3f}.",
        "Action":  "Use model scores to rank upcoming opportunities for resource allocation.",
        "Metric":  "Model-predicted vs actual participation, recalibrated each cohort."
    },
}

for code, rec in recommendations.items():
    print(f"\n{'='*60}")
    print(f"  {code}")
    print(f"  Finding: {rec['Finding']}")
    print(f"  Action:  {rec['Action']}")
    print(f"  KPI:     {rec['Metric']}")

# %% [markdown]
# ## 7. Measurement Framework for Future Cohorts
# %%
framework = pd.DataFrame({
    'KPI': [
        'Overall Participation Rate',
        'Rejection Rate',
        'Days Sign-up → Apply (Median)',
        'Top-5 Country Participation Rate',
        'Model AUC (recalibrated)',
        'Dropout Rate',
        'Event → Internship Conversion',
    ],
    'Baseline (Current)': [
        f"{df['is_participated'].mean():.1%}",
        f"{(df['Status Description']=='Rejected').mean():.1%}",
        f"{df['signup_to_apply_days'].median():.0f} days",
        'Varies by country',
        f"{summary['best_auc']:.3f}",
        f"{(df['Status Description']=='Dropped Out').mean():.1%}",
        'Not yet tracked',
    ],
    'Target': [
        '55%+', '<25%', '≤ 3 days',
        '60%+ for top 5', '≥ 0.90', '<5%', '≥ 20%'
    ],
    'Review Cadence': [
        'Each cohort','Each cohort','Monthly',
        'Quarterly','Each cohort','Each cohort','Annually'
    ]
})
print("\n=== MEASUREMENT FRAMEWORK ===")
print(framework.to_string(index=False))

# %% [markdown]
# ## 8. Reflection
# %%
reflection = """
INTERNSHIP REFLECTION – Week 4
================================
This internship provided deep, hands-on exposure to a real-world student
engagement dataset. Key personal takeaways:

1. DATA QUALITY MATTERS: Malformed timestamps and missing institution data
   required careful preprocessing — a reminder that clean pipelines are
   as important as the models themselves.

2. SIMPLE MODELS CAN SURPRISE: Logistic Regression achieved 0.878 AUC —
   strong for a baseline — while Gradient Boosting reached 0.917 with 
   careful tuning.

3. BUSINESS CONTEXT IS EVERYTHING: Without understanding what "participation"
   means programmatically, the model outputs would be meaningless.

4. COMMUNICATION IS THE FINAL MILE: The most important skill is translating
   model outputs into decisions program managers can act on.

Suggestions for future interns:
- Provide a codebook / data dictionary upfront.
- Include historical cohort labels to enable longitudinal modelling.
- Expose outcome data (did the opportunity lead to a job/internship?) 
  to measure downstream impact.
"""
print(reflection)

print("\n✅ Week 4 notebook complete — all outputs saved to outputs/figures/")
