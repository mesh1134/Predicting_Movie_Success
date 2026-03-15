<div align="center">
  <h1>🎬 Movie Success Predictor</h1>
</div>

## 📌 Overview
The **Movie Success Predictor** is a machine learning project designed to predict how successful a movie will be based on historical metadata. Originally built to classify movies into distinct success tiers (e.g., 'Flop', 'Average', 'Hit'), this project heavily relies on exploring variables like budgets, genres, and social media likes to forecast IMDb scores and broader reception.

## 🚀 What makes it interesting?
Unlike a standard single-model machine learning script, this project features an **optimized model election utility** (`mesh_utils_optimized.py`). 

Instead of just relying on a baseline algorithm, the codebase dynamically cross-evaluates multiple advanced classifiers (such as Random Forest, XGBoost, AdaBoost, SVM, and Logistic Regression) using `RandomizedSearchCV`. It automatically identifies the most accurate algorithm by analyzing the performance metrics against exactly the same data splits, ultimately exporting the true best-performing model for real-world deployment.

## 🛠️ Prerequisites
Before you begin, ensure you have the following installed on your local machine:
- **Python 3.8+**
- **pip** (Python package installer)

## 💻 Run Locally

1. **Clone the repository** (if you haven't already navigated to the folder):
   ```bash
   cd Predicting_Movie_Success
   ```

2. **Install all required dependencies** using the provided `requirements.txt` file (which includes pandas, scikit-learn, xgboost, etc.):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook or Script**:
   Experience the model training and evaluation by running the Jupyter Notebook `Movie_Predictor.ipynb`, or simply run the python script directly:
   ```bash
   python movie_success_predictor.py
   ```
