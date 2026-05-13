# %% [markdown]
# # Week 2 – Cohort Analysis & Exploratory Data Analysis
# **RIT Student Engagement Analysis**
#
# This notebook covers:
# 1. Feature engineering
# 2. Exploratory Data Analysis
# 3. Participation trend analysis
# 4. Cohort insights
# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

FIGDIR = '../outputs/figures'
CATS = ['#1B4F72','#2E86AB','#E84855','#F4A261','#2EC4B6','#8E44AD','#E67E22','#27AE60']

# %% [markdown]
# ## 1. Load & Feature Engineering
# %%
df = pd.read_csv('../data/raw/RIT_Opportunity_Wise_Data_Sheet.csv')
df.columns = df.columns.str.strip()
for col in ['Learner SignUp DateTime','Opportunity End Date','Entry created at',
            'Apply Date','Opportunity Start Date','Date of Birth']:
    df[col] = pd.to_datetime(df[col], errors='coerce')
df['Institution Name'].fillna('Unknown', inplace=True)
df['Current/Intended Major'].fillna('Unknown', inplace=True)

# Derived features
df['signup_to_apply_days'] = (df['Apply Date'] - df['Learner SignUp DateTime']).dt.days
df['age'] = (pd.Timestamp('2024-01-01') - df['Date of Birth']).dt.days / 365.25
df['age'] = df['age'].clip(15, 60)
df['signup_month'] = df['Learner SignUp DateTime'].dt.month
df['signup_year']  = df['Learner SignUp DateTime'].dt.year
df['is_participated'] = df['Status Description'].isin(
    ['Team Allocated','Started','Rewards Award']).astype(int)

print("Feature engineering complete. Shape:", df.shape)
print(f"Overall participation rate: {df['is_participated'].mean():.1%}")

# %% [markdown]
# ## 2. Sign-ups Over Time
# %%
monthly = df.groupby(df['Learner SignUp DateTime'].dt.to_period('M')).size()
monthly.index = monthly.index.to_timestamp()
fig, ax = plt.subplots(figsize=(12,5))
ax.fill_between(monthly.index, monthly.values, alpha=0.3, color='#2E86AB')
ax.plot(monthly.index, monthly.values, color='#1B4F72', linewidth=2)
ax.set_title('Monthly Student Sign-ups Over Time')
ax.set_xlabel('Month'); ax.set_ylabel('Sign-ups')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/06_signups_over_time.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. Participation Rate by Category
# %%
cat_part = df.groupby('Opportunity Category')['is_participated'].mean().sort_values(ascending=False)
print("Participation rates by category:")
print((cat_part*100).round(1).astype(str) + '%')

fig, ax = plt.subplots(figsize=(9,5))
bars = ax.bar(cat_part.index, cat_part.values*100, color=CATS[:len(cat_part)])
ax.set_title('Participation Rate by Opportunity Category')
ax.set_ylabel('Participation Rate (%)')
for bar, val in zip(bars, cat_part.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val*100:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/07_participation_by_category.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. Participation Rate by Country (Top 8)
# %%
top8 = df['Country'].value_counts().head(8).index
c8 = df[df['Country'].isin(top8)].groupby('Country')['is_participated'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(c8.index, c8.values*100, color=CATS[:len(c8)])
ax.set_title('Participation Rate by Country (Top 8)')
ax.set_ylabel('Participation Rate (%)'); ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/08_participation_by_country.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Heatmap – Category × Country
# %%
top5c = df['Country'].value_counts().head(5).index
heat_df = df[df['Country'].isin(top5c)].pivot_table(
    index='Opportunity Category', columns='Country',
    values='is_participated', aggfunc='mean')
fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(heat_df*100, annot=True, fmt='.1f', cmap='Blues',
            ax=ax, linewidths=0.5, cbar_kws={'label':'Participation Rate (%)'})
ax.set_title('Participation Rate (%) — Category × Country')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/11_heatmap_category_country.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Save Processed Dataset
# %%
df.to_csv('../data/processed/RIT_Cleaned_Engineered.csv', index=False)
print("✅ Processed dataset saved to data/processed/RIT_Cleaned_Engineered.csv")
