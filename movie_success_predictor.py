# Auto-generated from Movie_Predictor.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.append('./ml_utils')
from mesh_utils_optimized import find_best_model
from sklearn.metrics import accuracy_score


print("Loading dataset...")
df = pd.read_csv("movie_metadata.csv")
print(f"Dataset Shape: {df.shape}")
df.head() # Displays the first 5 rows to ensure it loaded correctly

print("Cleaning data and handling missing values...")

# Drop rows where the target variable is missing
df = df.dropna(subset=['imdb_score'])

# Drop columns that offer no predictive value
if 'movie_imdb_link' in df.columns:
    df = df.drop(columns=['movie_imdb_link'])

# Fill missing numericals with median, categoricals with mode/Unknown
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    if col in ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'plot_keywords']:
        df[col] = df[col].fillna('Unknown')
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Creating target variable 'Classify'...")
bins = [0, 3, 6, 10]
labels = ['Flop', 'Average', 'Hit']
df['Classify'] = pd.cut(df['imdb_score'], bins=bins, labels=labels, include_lowest=True)

# CRITICAL: Drop the imdb_score so the model doesn't cheat!
df = df.drop(columns=['imdb_score'])
print("Preprocessing complete. IMDB score removed.")

print("Addressing multicollinearity...")
# Find and drop highly correlated numerical features
num_features = df.select_dtypes(include=[np.number])
corr_matrix = num_features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
print(f"Dropping highly correlated columns: {to_drop}")
df = df.drop(columns=to_drop)

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
print("Data split successfully.")

print("--- Model 1: Logistic Regression (Baseline) ---")
log_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
print(classification_report(y_test, log_preds))

print("\n--- Model 2: Random Forest Classifier ---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print(classification_report(y_test, rf_preds))

print("Generating Confusion Matrix Visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Logistic Regression Confusion Matrix
sns.heatmap(confusion_matrix(y_test, log_preds), annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=labels, yticklabels=labels)
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Plot Random Forest Confusion Matrix
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Greens', ax=axes[1], 
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.show()

print("\n--- Model 3: Finding Best Model using mesh_utils_optimized ---")
import sys
sys.path.append('./ml_utils')
import mesh_utils_optimized
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Encode y for XGBoost which requires integer labels [0, 1, 2] instead of strings
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Fix the LogisticRegression instance in mesh_utils_optimized to support multiclass
# 'liblinear' does not support multiclass natively without OneVsRest. 'lbfgs' does.
lr_model = mesh_utils_optimized.classification_models["Logistic Regression"][0]
lr_model.set_params(solver='lbfgs', max_iter=2000)

# find_best_model does its own train_test_split internally with the same seed
best_model_results = mesh_utils_optimized.find_best_model(X_scaled, y_encoded, problem_type='classification', metric='accuracy', n_iter=5)

print(f"\n✓ Best Model found: {best_model_results['best_model_name']}")
print(f"✓ CV Accuracy: {best_model_results['CV_score']:.4f}")
print(f"✓ Test Accuracy: {best_model_results['Test_score']:.4f}")
print(f"✓ Best Parameters: {best_model_results['best_params']}")

# Compare Random Forest test accuracy to the best model test accuracy
rf_test_acc = accuracy_score(y_test, rf_preds)
print(f"\n--- Comparison ---")
print(f"Random Forest Test Accuracy: {rf_test_acc:.4f}")
print(f"Best Model ({best_model_results['best_model_name']}) Test Accuracy: {best_model_results['Test_score']:.4f}")
if best_model_results['Test_score'] > rf_test_acc:
    print("\nThe model found by mesh_utils is more accurate than the baseline Random Forest!")
else:
    print("\nThe baseline Random Forest is equal to or more accurate than the model found by mesh_utils.")


import joblib
import os

# 1. Create the Models directory if it doesn't exist yet
os.makedirs('../Models', exist_ok=True)

# 2. Export the trained Random Forest model to that folder
model_path = '../Models/rf_movie_model.pkl'
joblib.dump(rf_model, model_path)

print(f"✅ Success! Your model has been saved to: {model_path}")

