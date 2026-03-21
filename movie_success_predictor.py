# Auto-generated from Movie_Predictor.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set plot aesthetics — presentation-friendly settings
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["figure.facecolor"] = "white"
sns.set_style("whitegrid")


df = pd.read_csv("movie_metadata.csv")
print(f"Dataset Shape: {df.shape}")


# Dataset basic information
df.info()


# Summary statistics for numerical features
df.describe().T


# Summary statistics for categorical features
df.describe(include='object').T


# Check missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct})
print("\nMissing Values Summary:")
print(missing_df[missing_df['Missing Count'] > 0])


# Distribution of the target variable: IMDB Score
plt.figure(figsize=(12, 5))
sns.histplot(df['imdb_score'].dropna(), bins=30, kde=True, color='#6C5B7B')
plt.title('Distribution of IMDB Scores', fontsize=18, fontweight='bold')
plt.xlabel('IMDB Score', fontsize=13)
plt.ylabel('Frequency', fontsize=13)
plt.tight_layout()


# Univariate analysis for key numerical features
num_features_to_plot = ['duration', 'budget', 'gross', 'num_voted_users',
                        'num_user_for_reviews', 'num_critic_for_reviews',
                        'movie_facebook_likes', 'cast_total_facebook_likes']

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 18))
axes = axes.flatten()

for i, col in enumerate(num_features_to_plot):
    if col in df.columns:
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[i], color="#355C7D")
        axes[i].set_title(f"Distribution of {col}", fontsize=14, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=12)

plt.tight_layout()


# Univariate analysis for key categorical features
cat_features_to_plot = ['color', 'content_rating', 'language', 'country']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(cat_features_to_plot):
    if col in df.columns:
        top_vals = df[col].value_counts().head(10)
        sns.barplot(x=top_vals.values, y=top_vals.index, ax=axes[i], palette="viridis",
                    hue=top_vals.index, dodge=False, legend=False)
        axes[i].set_title(f"Top 10 - {col}", fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Count", fontsize=12)

plt.tight_layout()


# Create temporary Classify column for bivariate analysis
temp_df = df.copy()
temp_df['Classify'] = pd.cut(temp_df['imdb_score'], bins=[0, 3, 6, 10],
                              labels=['Flop', 'Average', 'Hit'], include_lowest=True)

# Numerical features vs Target (boxplots)
bivariate_num = ['duration', 'budget', 'gross', 'num_voted_users',
                 'num_critic_for_reviews', 'movie_facebook_likes']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(bivariate_num):
    if col in temp_df.columns:
        sns.boxplot(data=temp_df, x='Classify', y=col, ax=axes[i],
                    order=['Flop', 'Average', 'Hit'], palette='Set2',
                    hue='Classify', legend=False)
        axes[i].set_title(f"{col} by Movie Success", fontsize=14, fontweight="bold")

plt.tight_layout()


# Categorical features vs Target (stacked proportions)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, col in enumerate(['content_rating', 'color']):
    if col in temp_df.columns:
        ct = pd.crosstab(temp_df[col], temp_df['Classify'], normalize='index')
        # Keep only top categories
        if len(ct) > 8:
            top_idx = temp_df[col].value_counts().head(8).index
            ct = ct.loc[ct.index.isin(top_idx)]
        ct[['Flop', 'Average', 'Hit']].plot(kind='barh', stacked=True,
                                             ax=axes[i], colormap='RdYlGn')
        axes[i].set_title(f'Success Proportion by {col}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Proportion', fontsize=12)
        axes[i].legend(title='Category')

plt.tight_layout()
plt.show()



# Correlation heatmap of numerical features
plt.figure(figsize=(16, 10))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5, center=0)
plt.title('Correlation Heatmap of Numerical Features', fontsize=18, fontweight='bold')
plt.tight_layout()


# Drop rows where the target variable is missing
df = df.dropna(subset=['imdb_score'])

# Drop columns that offer no predictive value
if 'movie_imdb_link' in df.columns:
    df = df.drop(columns=['movie_imdb_link'])

# Fill missing numericals with median, categoricals with mode/Unknown
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df.loc[:, col] = df[col].fillna(df[col].median())
for col in cat_cols:
    if col in ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'plot_keywords']:
        df.loc[:, col] = df[col].fillna('Unknown')
    else:
        df.loc[:, col] = df[col].fillna(df[col].mode()[0])

print(f"Missing values after treatment: {df.isnull().sum().sum()}")


bins = [0, 3, 6, 10]
labels = ['Flop', 'Average', 'Hit']
df['Classify'] = pd.cut(df['imdb_score'], bins=bins, labels=labels, include_lowest=True)

# Visualize target distribution
plt.figure(figsize=(12, 5))
sns.countplot(data=df, x='Classify', hue='Classify', palette='viridis',
              order=['Flop', 'Average', 'Hit'], legend=False)
plt.title('Movie Success Classification Distribution', fontsize=18, fontweight='bold')
plt.xlabel('Success Category', fontsize=13)
plt.ylabel('Count', fontsize=13)
plt.show()

# Drop imdb_score to prevent data leakage
df = df.drop(columns=['imdb_score'])


# Identify continuous numerical features for outlier treatment
continuous_features = ['duration', 'num_critic_for_reviews', 'num_user_for_reviews',
                       'num_voted_users', 'movie_facebook_likes',
                       'director_facebook_likes', 'actor_1_facebook_likes',
                       'actor_2_facebook_likes', 'actor_3_facebook_likes',
                       'cast_total_facebook_likes', 'facenumber_in_poster']
# Only consider columns that actually exist in df
continuous_features = [c for c in continuous_features if c in df.columns]
# Detect outliers using IQR
Q1 = df[continuous_features].quantile(0.25)
Q3 = df[continuous_features].quantile(0.75)
IQR = Q3 - Q1
outlier_counts = ((df[continuous_features] < (Q1 - 1.5 * IQR)) | (df[continuous_features] > (Q3 + 1.5 * IQR))).sum()
print("Outlier counts per feature:")
print(outlier_counts[outlier_counts > 0].sort_values(ascending=False))
# Outliers are kept as-is because tree-based models (RF, XGBoost) use them effectively.
print("\nOutliers detected. We will NOT cap them to preserve valuable information for our tree-based models.")


# Find and drop highly correlated numerical features (threshold > 0.80)
num_features = df.select_dtypes(include=[np.number])
corr_matrix = num_features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
print(f"Dropping highly correlated columns: {to_drop}")


# Prepare X (features) and y (target)
X = df.drop(columns=['Classify'])
y = df['Classify']

# Label Encode categorical features
print("Encoding categorical variables...")
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into Train and Test sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples")


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluates model on train and test data, returns a summary DataFrame row."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    from sklearn.metrics import f1_score
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    
    print(f"\n--- {model_name} ---")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")
    print(f"Test F1 Score:     {test_f1:.4f}")
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    
    return pd.DataFrame({
        "Model": [model_name],
        "Train Accuracy": [round(train_acc, 4)],
        "Test Accuracy": [round(test_acc, 4)],
        "Test F1 Score": [round(test_f1, 4)]
    })


log_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
log_eval = evaluate_model(log_model, X_train, X_test, y_train, y_test, "Logistic Regression")

# Confusion Matrix
plt.figure(figsize=(8, 5))
cm_log = confusion_matrix(y_test, log_preds, labels=['Flop', 'Average', 'Hit'])
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 14},
            xticklabels=['Flop', 'Average', 'Hit'], yticklabels=['Flop', 'Average', 'Hit'])
plt.title('Logistic Regression - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual', fontsize=13)
plt.xlabel('Predicted', fontsize=13)
plt.tight_layout()


rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_eval = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")

# Confusion Matrix
plt.figure(figsize=(8, 5))
cm_rf = confusion_matrix(y_test, rf_preds, labels=['Flop', 'Average', 'Hit'])
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', annot_kws={'size': 14},
            xticklabels=['Flop', 'Average', 'Hit'], yticklabels=['Flop', 'Average', 'Hit'])
plt.title('Random Forest - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual', fontsize=13)
plt.xlabel('Predicted', fontsize=13)
plt.tight_layout()


# Advanced Hyperparameter Tuning & Model Selection
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Target Encoding
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Automated Model Hunt
import sys
sys.path.append('./ml_utils')
import mesh_utils_optimized
best_model_results = mesh_utils_optimized.find_best_model(X_scaled, y_encoded, problem_type='classification', metric='accuracy', n_iter=15)

# Validation Preparation
X_test_scaled = scaler.transform(X_test)
y_pred_best_numeric = best_model_results['trained_model'].predict(X_test_scaled)
y_pred_best = le_target.inverse_transform(y_pred_best_numeric)

print(f"\nFINAL ENGINE: {best_model_results['best_model_name'].upper()}")
print(f"Validation Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Validation F1 Score: {f1_score(y_test, y_pred_best, average='weighted'):.4f}")

# Cross-Model Performance Benchmarking

# Collecting Statistics
y_pred_mesh_num = best_model_results['trained_model'].predict(X_test_scaled)
best_model_test_preds = le_target.inverse_transform(y_pred_mesh_num)

mesh_eval = pd.DataFrame({
    "Model": [f"Mesh ({best_model_results['best_model_name']})"],
    "Train Accuracy": ["-"],
    "Test Accuracy": [round(accuracy_score(y_test, best_model_test_preds), 4)],
})

results = pd.concat([log_eval, rf_eval, mesh_eval], ignore_index=True)
print("\nBenchmarking Matrix:")
display(results)

# Visualizing Competitive Stakes
plt.figure(figsize=(12, 6))
models = results['Model'].tolist()
test_accs = [float(x) if x != '-' else 0 for x in results['Test Accuracy']]

sns.barplot(x=models, y=test_accs, palette='coolwarm')

for i, v in enumerate(test_accs):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

plt.title('Predictive Engine Showdown: Accuracy', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Test Accuracy', fontsize=13)
plt.ylim(0, 1.1)
plt.show()

# Feature Importance from Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Top 10 features
top_n = 10
top_indices = indices[:top_n]
top_features = [X.columns[i] for i in top_indices]
top_importances = importances[top_indices]

plt.figure(figsize=(12, 6))
sns.barplot(x=top_importances, y=top_features, hue=top_features,
            palette="rocket", dodge=False, legend=False)
plt.title('Top 10 Most Important Features Dictating Movie Success', fontsize=18, fontweight='bold')
plt.xlabel('Relative Importance', fontsize=13)
plt.ylabel('Feature', fontsize=13)
plt.tight_layout()


import joblib
import os
try:
    models_dir = './models'
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'best_movie_model.pkl')
    
    if 'best_model_results' in locals() and best_model_results.get('trained_model') is not None:
        joblib.dump(best_model_results['trained_model'], model_path)
        print(f"Best model ({best_model_results['best_model_name']}) saved to: {model_path}")
    elif 'rf_model' in locals():
        rf_path = os.path.join(models_dir, 'rf_movie_model.pkl')
        joblib.dump(rf_model, rf_path)
        print(f"Random Forest model saved to: {rf_path}")
    else:
        print("No trained models found to export.")
except Exception as e:
    print(f"An error occurred while exporting the model: {e}")


