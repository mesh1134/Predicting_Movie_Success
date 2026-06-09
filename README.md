<div align="center">
  <h1>🎬 Movie Success Predictor AI</h1>
  <p><b>An End-to-End Machine Learning Pipeline & Interactive Dashboard for Box Office Forecasting</b></p>
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-175591.svg)](https://xgboost.ai/)
</div>

<br/>

## 📌 Overview
The **Movie Success Predictor** is a comprehensive machine learning project designed to predict a movie's financial success ('Hit', 'Average', or 'Flop') before it hits the theaters. 

By analyzing historical metadata—ranging from budgets and genres to social media engagement (Facebook likes) and cast details—this project provides actionable insights into what makes a movie profitable. 

Originally built as an analytical notebook, the project has evolved into a fully interactive, production-ready **Streamlit web application** featuring a sleek dark-mode UI, single-prediction engines, and batch dataset analysis dashboards.

## ✨ Key Features

### 🧠 Advanced Machine Learning Pipeline
- **Optimized Model Selection:** Features a custom utility (`ml_utils/mesh_utils_optimized.py`) that dynamically cross-evaluates multiple advanced classifiers (Random Forest, XGBoost, AdaBoost, SVM, and Logistic Regression) using `RandomizedSearchCV`.
- **Automated Deployment:** It automatically identifies the most accurate algorithm by analyzing performance metrics against identical data splits, exporting the true best-performing model (currently **XGBoost**) for real-world deployment.

### 💻 Interactive Streamlit Application
- **Single Movie Prediction Engine:** Select an existing movie or enter hypothetical production details (budget, duration, cast likes, etc.) to simulate potential box office outcomes.
- **Batch Dataset Analysis:** Upload a CSV file of upcoming movies to generate predictions in bulk. The app gracefully handles missing data and unseen categories.
- **Confidence Analysis:** Explains the model's certainty with probability breakdowns for each success tier.

### 📊 Rich Analytical Dashboards (Plotly)
- **Market Positioning:** Visualizes where your movie stands against historical data (Budget vs. Gross Landscape).
- **Financial & Genre Insights:** Explores ROI distribution and genre success rates via interactive heatmaps and scatter plots.
- **Downloadable Reports:** Export batch prediction results directly to CSV.

## 🛠️ Technology Stack
* **Data Processing & ML:** Python, Pandas, Scikit-Learn, XGBoost, Joblib
* **Web Application:** Streamlit
* **Data Visualization:** Plotly (Express & Graph Objects)
* **Environment:** Jupyter Notebooks, pip

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+**
- **pip** (Python package installer)

### Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Predicting_Movie_Success.git
   cd Predicting_Movie_Success
   ```

2. **Install all required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit App**:
   Experience the premium interactive UI by running:
   ```bash
   streamlit run app.py
   ```

4. **Explore the Machine Learning Process**:
   To see the data cleaning, feature engineering, and model training steps, open the Jupyter Notebook:
   ```bash
   jupyter notebook Movie_Predictor_latest.ipynb
   ```

## 📁 Repository Structure
* `app.py`: The main Streamlit web application.
* `Movie_Predictor_latest.ipynb`: The primary notebook containing data exploration, visualization, and model training.
* `ml_utils/mesh_utils_optimized.py`: The core script for hyperparameter tuning and model evaluation.
* `models/`: Serialized ML models, scalers, and encoders.
* `cleaned_movie_data.csv`: The processed dataset powering the application.
* `requirements.txt`: Python dependencies.

## 💡 Why This Project?
This app demonstrates the complete lifecycle of a Machine Learning project:
1. **Data Engineering:** Handling messy real-world datasets, imputing missing values, and engineering new features.
2. **Model Optimization:** Rigorously testing multiple algorithms to find the global optimum rather than settling for a baseline.
3. **Software Engineering:** Packaging the predictive model into an accessible, robust, and visually captivating web application designed for end-users and stakeholders.

---
*If you find this project interesting, feel free to ⭐ the repository or connect with me on LinkedIn! - https://www.linkedin.com/in/soumesh-satheesan-46b287261/*
