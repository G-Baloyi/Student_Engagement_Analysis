# %% [markdown]
# # Week 1 – Data Understanding Report
# **RIT Student Engagement Analysis**
# 
# This notebook covers:
# 1. Dataset overview
# 2. Data dictionary
# 3. Initial data preparation
# 4. Summary statistics
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

FIGDIR = '../outputs/figures'
PALETTE = ['#1B4F72','#2E86AB','#E84855','#F4A261','#2EC4B6']

# %% [markdown]
# ## 1. Load Dataset
# %%
df = pd.read_csv('../data/raw/RIT_Opportunity_Wise_Data_Sheet.csv')
df.columns = df.columns.str.strip()
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
df.head(3)

# %% [markdown]
# ## 2. Data Dictionary
# %%
data_dict = {
    'Column': [
        'Learner SignUp DateTime', 'Opportunity Id', 'Opportunity Name',
        'Opportunity Category', 'Opportunity End Date', 'First Name',
        'Date of Birth', 'Gender', 'Country', 'Institution Name',
        'Current/Intended Major', 'Entry created at', 'Status Description',
        'Status Code', 'Apply Date', 'Opportunity Start Date'
    ],
    'Description': [
        'Timestamp when the student signed up for the platform',
        'Unique identifier for each opportunity',
        'Full name of the learning opportunity',
        'Type: Course, Internship, Event, Competition, Engagement',
        'Date and time the opportunity ends',
        'Student first name',
        'Student date of birth',
        'Student-reported gender',
        'Country of the student',
        'Name of the student\'s academic institution',
        'Student\'s current or intended field of study',
        'System timestamp when the record was created',
        'Descriptive participation status (e.g., Started, Rejected)',
        'Numeric code corresponding to Status Description',
        'Date the student applied to the opportunity',
        'Scheduled start date of the opportunity'
    ],
    'Data Type': [
        'datetime', 'string (UUID)', 'string', 'categorical',
        'datetime', 'string', 'datetime', 'categorical',
        'string', 'string', 'string', 'datetime',
        'categorical', 'integer', 'datetime', 'datetime'
    ]
}
dd = pd.DataFrame(data_dict)
print(dd.to_string(index=False))

# %% [markdown]
# ## 3. Initial Data Preparation
# %%
# Parse datetime columns
datetime_cols = ['Learner SignUp DateTime','Opportunity End Date','Entry created at',
                 'Apply Date','Opportunity Start Date','Date of Birth']
for col in datetime_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Fill missing
df['Institution Name'].fillna('Unknown', inplace=True)
df['Current/Intended Major'].fillna('Unknown', inplace=True)

print("Missing values after cleaning:")
print(df.isnull().sum())

# %% [markdown]
# ## 4. Summary Statistics
# %%
print("=== Overall Summary ===")
print(f"Total records:        {len(df):,}")
print(f"Unique opportunities: {df['Opportunity Name'].nunique()}")
print(f"Unique students:      {df['First Name'].nunique()}")
print(f"Date range:           {df['Learner SignUp DateTime'].min().date()} → {df['Learner SignUp DateTime'].max().date()}")
print(f"\nStatus breakdown:")
print(df['Status Description'].value_counts())
print(f"\nOpportunity categories:")
print(df['Opportunity Category'].value_counts())

# Plot status distribution
fig, ax = plt.subplots(figsize=(10,5))
counts = df['Status Description'].value_counts()
ax.barh(counts.index, counts.values, color=PALETTE*2)
ax.set_xlabel('Count'); ax.set_title('Student Status Distribution')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/01_status_distribution.png', bbox_inches='tight')
plt.show()
print("Figure saved.")
