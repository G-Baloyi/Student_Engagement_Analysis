# %% [markdown]
# # Week 3 – Machine Learning Modeling & Predictions
# **RIT Student Engagement Analysis**
#
# This notebook covers:
# 1. Feature preparation for ML
# 2. Model training (Logistic Regression, Random Forest, Gradient Boosting)
# 3. Model evaluation & comparison
# 4. Feature importance & predicted probabilities
# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

FIGDIR = '../outputs/figures'
PALETTE = {'primary':'#1B4F72','secondary':'#2E86AB','accent':'#E84855',
           'success':'#2EC4B6','muted':'#95A5A6'}
CATS = ['#1B4F72','#2E86AB','#E84855','#F4A261','#2EC4B6']

# %% [markdown]
# ## 1. Load Processed Data
# %%
df = pd.read_csv('../data/processed/RIT_Cleaned_Engineered.csv')
for col in ['Learner SignUp DateTime','Apply Date','Date of Birth']:
    df[col] = pd.to_datetime(df[col], errors='coerce')
print("Data loaded. Shape:", df.shape)

# %% [markdown]
# ## 2. Feature Engineering for ML
# %%
le_cat = LabelEncoder(); le_gen = LabelEncoder(); le_cty = LabelEncoder()
df['cat_enc']     = le_cat.fit_transform(df['Opportunity Category'].astype(str))
df['gender_enc']  = le_gen.fit_transform(df['Gender'].astype(str))
df['country_enc'] = le_cty.fit_transform(df['Country'].astype(str))

features = ['cat_enc','gender_enc','country_enc','age',
            'signup_to_apply_days','signup_month','signup_year']
X = df[features].copy()
y = df['is_participated']

imp = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imp.fit_transform(X), columns=features)

print(f"Features: {features}")
print(f"Class balance:\n{y.value_counts(normalize=True).round(3)}")

X_train,X_test,y_train,y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# %% [markdown]
# ## 3. Train Models
# %%
models = {
    'Logistic Regression': Pipeline([
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    yp  = model.predict(X_test)
    ypr = model.predict_proba(X_test)[:,1]
    cv  = cross_val_score(model, X_imp, y, cv=5, scoring='roc_auc')
    results[name] = {
        'model': model, 'y_pred': yp, 'y_proba': ypr,
        'auc': roc_auc_score(y_test, ypr), 'cv_auc': cv.mean(),
        'cm':  confusion_matrix(y_test, yp),
    }
    print(f"  Test AUC: {results[name]['auc']:.4f}  |  CV AUC: {results[name]['cv_auc']:.4f}")
    print(classification_report(y_test, yp, target_names=['Not Participated','Participated']))

best_name = max(results, key=lambda k: results[k]['auc'])
best = results[best_name]
print(f"\n🏆 Best Model: {best_name}  (AUC = {best['auc']:.4f})")

# %% [markdown]
# ## 4. Model Comparison – AUC
# %%
names = list(results.keys())
aucs = [results[n]['auc'] for n in names]
cv_aucs = [results[n]['cv_auc'] for n in names]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9,5))
ax.bar(x-0.2, aucs, 0.35, label='Test AUC', color=PALETTE['primary'])
ax.bar(x+0.2, cv_aucs, 0.35, label='CV AUC (5-fold)', color=PALETTE['secondary'])
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
ax.set_ylim(0.5,1.0); ax.set_title('Model Comparison – ROC-AUC Scores')
ax.set_ylabel('AUC Score'); ax.legend()
for i,(a,b) in enumerate(zip(aucs,cv_aucs)):
    ax.text(i-0.2, a+0.005, f'{a:.3f}', ha='center', fontsize=9)
    ax.text(i+0.2, b+0.005, f'{b:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/13_model_comparison_auc.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. ROC Curves
# %%
fig, ax = plt.subplots(figsize=(8,6))
colors = [PALETTE['primary'], PALETTE['accent'], PALETTE['success']]
for (name, res), col in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, label=f'{name} (AUC={res["auc"]:.3f})', color=col, linewidth=2)
ax.plot([0,1],[0,1],'--', color=PALETTE['muted'])
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves – All Models'); ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/14_roc_curves.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Confusion Matrix (Best Model)
# %%
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(best['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Not Participated','Participated'],
            yticklabels=['Not Participated','Participated'])
ax.set_title(f'Confusion Matrix – {best_name}')
ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/15_confusion_matrix.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Feature Importance
# %%
rf = results['Random Forest']['model']
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(fi.index, fi.values, color=CATS[:len(fi)])
ax.set_title('Feature Importance – Random Forest'); ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/16_feature_importance.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Predicted Probability Distribution
# %%
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(best['y_proba'][y_test==0], bins=30, alpha=0.6,
        color=PALETTE['accent'], label='Not Participated')
ax.hist(best['y_proba'][y_test==1], bins=30, alpha=0.6,
        color=PALETTE['success'], label='Participated')
ax.set_title(f'Predicted Participation Probability – {best_name}')
ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Count'); ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGDIR}/17_predicted_probabilities.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Save Predictions
# %%
df_test = df.iloc[X_test.index].copy()
df_test['predicted_probability'] = best['y_proba']
df_test['predicted_participated'] = best['y_pred']
df_test[['Opportunity Name','Opportunity Category','Status Description',
         'predicted_probability','predicted_participated']].to_csv(
    '../data/processed/predictions.csv', index=False)
print("✅ Predictions saved to data/processed/predictions.csv")
