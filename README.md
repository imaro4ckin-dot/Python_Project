# 🚀 Startup Profit Predictor & Analysis

A comprehensive, interactive web dashboard built with Streamlit and Python. This application performs deep exploratory data analysis (EDA), predicts startup profits using Multiple Linear Regression, and provides transparent statistical evidence to back up its predictions.

## 📖 About the Project

This tool is designed to help stakeholders, investors, and data enthusiasts understand which financial departments (R&D, Administration, Marketing) drive the most profit for startups. Using a dataset of 50 startups, the app trains a machine learning model to estimate future profits based on user-defined spending allocations.

## ✨ Features

The dashboard is divided into three intuitive tabs:

* **📊 Tab 1: Business Overview & EDA**
    * High-level KPI metrics (Average Profit, Top Performing Market).
    * Deep dive into "Zero-Spend Anomalies" showing the impact of cutting R&D or Marketing entirely.
    * Correlation heatmaps and impact score bar charts to identify the strongest profit drivers.
    * Regional performance breakdown across different states.
    * Expandable raw descriptive statistics.
* **🤖 Tab 2: Profit Prediction**
    * Interactive sliders to adjust R&D, Administration, and Marketing budgets.
    * State selection dropdown.
    * Real-time profit estimation using a trained Multiple Linear Regression model.
* **⚖️ Tab 3: Statistical Evidence**
    * Transparent model accuracy metrics (R² Score, MAE, RMSE).
    * Visual proof of accuracy (Actual vs. Predicted scatter plot).
    * Feature impact visualization showing the exact dollar contribution of each department.
    * Advanced error analysis, including residual scatter plots and error distribution histograms to prove model reliability.

## 🛠️ Tech Stack

* **Python 3.x**
* **Streamlit:** For building the interactive web interface.
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Scikit-Learn:** For training the Linear Regression model and calculating metrics.
* **Matplotlib & Seaborn:** For creating high-quality statistical visualizations.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. You will also need to install the required libraries. You can do this by running:

```bash
pip install streamlit pandas numpy seaborn matplotlib scikit-learn
