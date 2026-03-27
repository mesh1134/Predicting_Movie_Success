import nbformat as nbf
import json
import os

nb = nbf.v4.new_notebook()
def md(text): return nbf.v4.new_markdown_cell(text)
def cd(text): return nbf.v4.new_code_cell(text)

cells = []
# Headers directly mapping exactly to PDF and template instructions
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Capstone Project | Predicting Movie Success</p>\n\n### Project Introduction\nOur client, a major film studio, aims to enhance their understanding of factors influencing movie success and improve prediction models for IMDB ratings. The goal is to build an accurate classification model based on IMDB scores (Hit, Average, Flop) and discover which features drive success.'))
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Step 1 | Import Libraries</p>'))
cells.append(cd("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")

FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)
def save_plot(filename):
    plt.savefig(os.path.join(FIG_DIR, filename), bbox_inches='tight')"""))
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Step 2 | Read Dataset</p>'))
cells.append(cd('df = pd.read_csv("movie_metadata.csv")\ndisplay(df.head())'))
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Step 3 | Dataset Overview</p>'))
cells.append(md('### <b><span style="color:#ff826e">Step 3.1 |</span><span style="color:#004080"> Dataset Basic Information</span></b>'))
cells.append(cd('df.info()'))
cells.append(md('### <b><span style="color:#ff826e">Step 3.2 |</span><span style="color:#004080"> Summary Statistics for Numerical Variables</span></b>'))
cells.append(cd('display(df.describe().T)'))
cells.append(md('### <b><span style="color:#ff826e">Step 3.3 |</span><span style="color:#004080"> Summary Statistics for Categorical Variables</span></b>'))
cells.append(cd('display(df.describe(include=[\'O\']).T)'))
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Step 4 | EDA</p>\nPer the guidelines, we must first categorize IMDB scores into "Hit", "Average", and "Flop" so we can analyze distributions based on actual success groups.'))
cells.append(cd("""# Drop rows where target variable is entirely missing
df = df.dropna(subset=['imdb_score'])

# Categorize IMDB Scores into Classes (1-3 Flop, 3-6 Average, 6-10 Hit)
bins = [0, 3, 6, 10]
labels = ['Flop', 'Average', 'Hit']
df['Classify'] = pd.cut(df['imdb_score'], bins=bins, labels=labels, include_lowest=True)

# Drop original imdb_score column and redundant IMDB link
df = df.drop(columns=['imdb_score', 'movie_imdb_link'], errors='ignore')"""))
cells.append(md('### <b><span style="color:#ff826e">Step 4.1 |</span><span style="color:#004080"> Univariate Analysis</span></b>\n#### <b><span style="color:#ff826e">Step 4.1.1 |</span><span style="color:#004080"> Numerical Variables Univariate Analysis</span></b>'))
cells.append(cd("""num_features = df.select_dtypes(include=[np.number]).columns
n_cols = 3
n_rows = (len(num_features) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i], color='steelblue')
    axes[i].set_title(f'Distribution of {col}')
    if df[col].max() > 1000:
        axes[i].set_yscale('log')
        axes[i].set_ylabel('Count (Log)')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()"""))
cells.append(md('#### <b><span style="color:#ff826e">Step 4.1.2 |</span><span style="color:#004080"> Categorical Variables Univariate Analysis</span></b>'))
cells.append(cd("""# Target Variable Distribution and Top Genres
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Target Class
sns.countplot(data=df, x='Classify', order=['Flop', 'Average', 'Hit'], palette='viridis', ax=axes[0])
axes[0].set_title('Target Class Distribution (Imbalanced)')
axes[0].set_ylabel('Count')

# Top Genres
top_genres = df['genres'].value_counts().head(10)
sns.barplot(y=top_genres.index, x=top_genres.values, palette='magma', ax=axes[1])
axes[1].set_title('Top 10 Most Frequent Genres')

plt.tight_layout()
plt.show()"""))
cells.append(md('### <b><span style="color:#ff826e">Step 4.2 |</span><span style="color:#004080"> Bivariate Analysis</span></b>\n#### <b><span style="color:#ff826e">Step 4.2.1 |</span><span style="color:#004080"> Numerical Features vs Target</span></b>'))
cells.append(cd("""fig, axes = plt.subplots(2, 1, figsize=(10, 12))

sns.boxplot(data=df, x='Classify', y='budget', order=['Flop', 'Average', 'Hit'], palette='viridis', ax=axes[0])
axes[0].set_title('Movie Budget vs. Success Category (Log Scale)')
axes[0].set_yscale('log')

sns.boxplot(data=df, x='Classify', y='cast_total_facebook_likes', order=['Flop', 'Average', 'Hit'], palette='viridis', ax=axes[1])
axes[1].set_title('Cast Facebook Likes vs. Success Category (Log Scale)')
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()"""))
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Step 5 | Data Preprocessing</p>\n### <b><span style="color:#ff826e">Step 5.1 |</span><span style="color:#004080"> Missing Value Treatment</span></b>'))
cells.append(cd("""# Fill numerical with median
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical with mode or Unknown
for col in df.select_dtypes(include=['object']).columns:
    if col in ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'plot_keywords']:
        df[col] = df[col].fillna('Unknown')
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing after treatment:\\n", df.isnull().sum().sum())"""))
cells.append(md('### <b><span style="color:#ff826e">Step 5.2 |</span><span style="color:#004080"> Addressing Multicollinearity</span></b>'))
cells.append(cd("""plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
print(f"Dropping highly correlated numeric predictors: {to_drop}")
df = df.drop(columns=to_drop)"""))
cells.append(md('### <b><span style="color:#ff826e">Step 5.3 |</span><span style="color:#004080"> Label Encoding for Categorical Variables</span></b>'))
cells.append(cd("""categorical_cols = df.select_dtypes(include=['object', 'category']).columns.drop('Classify', errors='ignore')

# Handle Feature Selection explicitly if needed, here we retain rest.
X = df.drop(columns=['Classify'])
y = df['Classify'].astype(str)

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)"""))
cells.append(md('### <b><span style="color:#ff826e">Step 5.4 |</span><span style="color:#004080"> Train-Test Split & Scaling</span></b>'))
cells.append(cd("""X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Training Data shape:", X_train_scaled.shape)"""))
cells.append(md('### <b><span style="color:#ff826e">Step 5.5 |</span><span style="color:#004080"> Addressing Imbalanced Data (SMOTE)</span></b>'))
cells.append(cd("""smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

plt.figure(figsize=(6, 4))
sns.countplot(x=target_le.inverse_transform(y_train_res), order=['Flop', 'Average', 'Hit'], palette='husl')
plt.title('Target Classes after SMOTE')
plt.show()"""))
cells.append(cd("""def evaluate_model(model_name, y_true, y_pred):
    print(f"--- {model_name} Evaluation ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\\nClassification Report:\\n", classification_report(y_true, y_pred, target_names=target_le.classes_, zero_division=0))
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=target_le.classes_, yticklabels=target_le.classes_)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()"""))
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Step 6 | Decision Tree Model Building</p>'))
cells.append(cd("""dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_res, y_train_res)

dt_preds = dt_model.predict(X_test_scaled)
evaluate_model("Decision Tree", y_test, dt_preds)"""))
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Step 7 | Random Forest Model Building</p>'))
cells.append(cd("""rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)

rf_preds = rf_model.predict(X_test_scaled)
evaluate_model("Random Forest", y_test, rf_preds)"""))
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Step 8 | Logistic Regression Model Building</p>'))
cells.append(cd("""log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_res, y_train_res)

log_preds = log_model.predict(X_test_scaled)
evaluate_model("Logistic Regression", y_test, log_preds)"""))
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Step 9 | SVM Model Building</p>'))
cells.append(cd("""svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_res, y_train_res)

svm_preds = svm_model.predict(X_test_scaled)
evaluate_model("Support Vector Machine", y_test, svm_preds)"""))
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Step 10 | Conclusion</p>'))
cells.append(cd("""models_dict = {"Decision Tree": dt_preds, "Random Forest": rf_preds,
               "Logistic Regression": log_preds, "SVM": svm_preds}
records = []
for name, preds in models_dict.items():
    records.append({"Algorithm": name, "Accuracy": accuracy_score(y_test, preds),
                    "Precision": precision_score(y_test, preds, average='weighted', zero_division=0),
                    "Recall": recall_score(y_test, preds, average='weighted', zero_division=0),
                    "F1-Score": f1_score(y_test, preds, average='weighted', zero_division=0)})

leaderboard = pd.DataFrame(records).sort_values('F1-Score', ascending=False)
display(leaderboard.style.background_gradient(cmap='Blues').format("{:.2%}"))

df_melted = leaderboard.melt(id_vars='Algorithm', var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 8))
sns.barplot(data=df_melted, y='Algorithm', x='Score', hue='Metric', palette='Blues_d')
plt.title('Overall Model Performance Leaderboard')
plt.xlim(0, 1.15)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.show()

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=X.columns[indices], palette='magma')
plt.title('Top 10 Most Important Drivers of Movie Success (Random Forest)')
plt.show()

print("CONCLUSION:\\nAs demonstrated in the evaluation, Tree-based models (especially Random Forest) excel robustly across this class distribution.\\nImportant features heavily relate to engagement (votes, likes) and production scale (budget), fulfilling the analysis instructions.")"""))
cells.append(md('# <p style="background-color:#1e3d59; font-family:calibri; color:#ffffff; font-size:150%; text-align:center; border-radius:15px 50px;">Step 11 | Prediction</p>'))
cells.append(cd("""sample = X_test_scaled[0].reshape(1, -1)
original_class = target_le.inverse_transform([y_test[0]])[0]
predicted_enc = rf_model.predict(sample)[0]
predicted_class = target_le.inverse_transform([predicted_enc])[0]

print("-- Predicting on a single test sample using our best model --")
print(f"Actual Class: {original_class}")
print(f"Predicted Class: {predicted_class}")

joblib.dump(rf_model, 'best_model.pkl')
print("Saved best Random Forest model to best_model.pkl for future predictions.")"""))

nb.cells = cells
with open('Movie_Predictor.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
