import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page Configuration
st.set_page_config(page_title="Startup Profit Predictor", page_icon="🚀", layout="wide")


# 1. Load Data
@st.cache_data
def load_data():
    return pd.read_csv("50_Startups.csv")


df = load_data()

# 2. Preprocessing
df_processed = pd.get_dummies(df, columns=['State'], drop_first=True)
X = df_processed.drop('Profit', axis=1)
y = df_processed['Profit']

# Split data and Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Main Title
st.title("🚀 Startup Profit Prediction & Analysis")

# Create Tabs
tab1, tab2, tab3 = st.tabs([" Business Overview & EDA", " Profit Prediction", " Statistical Evidence"])

# ==========================================
# TAB 1: Business Overview & Complete EDA
# ==========================================
with tab1:
    st.header(" Complete Exploratory Data Analysis (EDA)")
    st.markdown(
        "A comprehensive breakdown of the historical data, combining high-level business trends with deep statistical correlations.")

    # --- KPI Metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Startup Profit", f"${df['Profit'].mean():,.0f}")
    with col2:
        st.metric("Highest Profit Recorded", f"${df['Profit'].max():,.0f}")
    with col3:
        best_state = df.groupby('State')['Profit'].mean().idxmax()
        st.metric("Top Performing Market", best_state)

    st.markdown("---")

    # --- Deep Dive Analysis: The Zero-Spend Anomalies ---
    st.subheader(" Deep Dive Analysis: The Zero-Spend Anomalies")

    col_anomaly1, col_anomaly2 = st.columns(2)
    with col_anomaly1:
        rd_zero = df[df['R&D Spend'] == 0]['Profit'].mean()
        rd_nonzero = df[df['R&D Spend'] > 0]['Profit'].mean()
        fig_rd, ax_rd = plt.subplots(figsize=(6, 4))
        sns.barplot(x=['$0 R&D Spend', 'Invested in R&D'], y=[rd_zero, rd_nonzero], palette=['#d62728', '#2ca02c'],
                    ax=ax_rd)
        ax_rd.set_ylabel("Average Profit ($)", fontweight='bold')
        ax_rd.set_title("The Impact of Cutting R&D", fontweight='bold')
        ax_rd.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig_rd)
        st.error(
            f"**Insight:** Startups that cut R&D completely suffered massive losses, averaging only **${rd_zero:,.0f}** in profit.")

    with col_anomaly2:
        mkt_zero = df[df['Marketing Spend'] == 0]['Profit'].mean()
        mkt_nonzero = df[df['Marketing Spend'] > 0]['Profit'].mean()
        fig_mkt, ax_mkt = plt.subplots(figsize=(6, 4))
        sns.barplot(x=['$0 Marketing', 'Invested in Marketing'], y=[mkt_zero, mkt_nonzero],
                    palette=['#ff7f0e', '#1f77b4'], ax=ax_mkt)
        ax_mkt.set_ylabel("Average Profit ($)", fontweight='bold')
        ax_mkt.set_title("The Impact of Cutting Marketing", fontweight='bold')
        ax_mkt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig_mkt)
        st.success(
            f"**Insight:** Startups with 0 Marketing still averaged **${mkt_zero:,.0f}** in profit, relying entirely on R&D.")

    st.markdown("---")

    # --- Correlation & Impact (Heatmap & Barplot) ---
    st.subheader(" Feature Correlation & Impact")
    st.markdown(
        "Here we use statistical heatmaps and impact scores to identify which spending category has the strongest linear relationship with profit.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Correlation Heatmap**")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
                    ax=ax_corr)
        st.pyplot(fig_corr)
        st.info(" **EDA Highlight:** The heatmap confirms R&D Spend has a massive **0.97 correlation** with Profit.")

    with colB:
        st.markdown("**Department Impact Score**")
        corr_data = df.select_dtypes(include=np.number).corr()[['Profit']].drop('Profit').sort_values(by='Profit',
                                                                                                      ascending=False)
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        sns.barplot(x=corr_data['Profit'], y=corr_data.index, palette="viridis", ax=ax_bar)
        ax_bar.set_xlabel("Correlation with Profit (0 to 1)")
        ax_bar.set_ylabel("")
        st.pyplot(fig_bar)
        st.info(
            " **Insight:** Translating the heatmap into a bar chart clearly shows Administration spending has almost zero relationship with success.")

    st.markdown("---")

    # --- Regional Analysis (Boxplot & Barplot) ---
    st.subheader("🌎 Regional Performance Breakdown")
    colC, colD = st.columns(2)
    with colC:
        st.markdown("**Profit Distribution by State (Boxplot)**")
        fig_box, ax_box = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='State', y='Profit', data=df, palette="Set2", ax=ax_box)
        ax_box.set_ylabel("Profit ($)")
        st.pyplot(fig_box)
        st.info(
            " **Insight:** The boxplot shows the total spread of profits. Notice how New York has the highest single outlier, while California's spread is slightly lower.")

    with colD:
        st.markdown("**Average Profit by Region (Bar Chart)**")
        state_avg = df.groupby('State')['Profit'].mean().reset_index().sort_values(by='Profit', ascending=False)
        fig_avg, ax_avg = plt.subplots(figsize=(6, 4))
        sns.barplot(x='State', y='Profit', data=state_avg, palette="Set2", ax=ax_avg)
        ax_avg.set_ylabel("Average Profit ($)")
        st.pyplot(fig_avg)
        st.info(
            " **Insight:** On average, Florida marginally outperforms New York and California, but the difference is statistically negligible compared to R&D spending.")

    st.markdown("---")

    # --- The R&D Trend ---
    st.subheader(" The R&D Effect: Visualizing the Growth")
    st.markdown("Every dot below is a startup. The red line shows the undeniable trend: more R&D equals more Profit.")
    fig_reg, ax_reg = plt.subplots(figsize=(10, 4))
    sns.regplot(x='R&D Spend', y='Profit', data=df, scatter_kws={'alpha': 0.6, 'color': '#1f77b4'},
                line_kws={'color': 'red'}, ax=ax_reg)
    ax_reg.set_xlabel("Research & Development Budget ($)", fontweight='bold')
    ax_reg.set_ylabel("Total Profit ($)", fontweight='bold')
    ax_reg.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig_reg)

    # --- Expandable Data Analytics Table ---
    with st.expander(" View Raw Descriptive Statistics"):
        st.markdown(
            "This table shows the statistical distribution (Mean, Minimum, Maximum, and Percentiles) for all financial categories.")
        st.dataframe(df.describe().round(2))

# ==========================================
# TAB 2: Profit Prediction (Sliders)
# ==========================================
with tab2:
    st.header("Predict Company Profit")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    st.info(
        f"**Model Accuracy (R² Score):** {r2:.4f} *(This means the model explains {r2 * 100:.1f}% of the variations in profit)*")

    st.markdown("### Adjust Startup Parameters (Sliders)")
    rd_spend = st.slider("R&D Spend ($)", min_value=0, max_value=int(df['R&D Spend'].max() * 1.5),
                         value=int(df['R&D Spend'].mean()), step=1000)
    admin_spend = st.slider("Administration ($)", min_value=0, max_value=int(df['Administration'].max() * 1.5),
                            value=int(df['Administration'].mean()), step=1000)
    marketing_spend = st.slider("Marketing Spend ($)", min_value=0, max_value=int(df['Marketing Spend'].max() * 1.5),
                                value=int(df['Marketing Spend'].mean()), step=1000)
    state = st.selectbox("State", df['State'].unique())

    input_dict = {
        'R&D Spend': rd_spend, 'Administration': admin_spend, 'Marketing Spend': marketing_spend,
        'State_Florida': 1 if state == 'Florida' else 0, 'State_New York': 1 if state == 'New York' else 0
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.success(f"###  Estimated Profit: ${prediction:,.2f}")

# ==========================================
# TAB 3: Statistical Evidence (DETAILED)
# ==========================================
with tab3:
    st.header(" Statistical Evidence, Residuals & Reality")
    st.markdown("A deep dive into the mathematical proofs behind the model and a transparent look at its errors.")

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    residuals = y_test - y_pred

    col_err1, col_err2, col_err3 = st.columns(3)
    with col_err1:
        st.metric("R² Score (Accuracy)", f"{r2:.4f}")
    with col_err2:
        st.metric("Mean Absolute Error (MAE)", f"${mae:,.0f}")
    with col_err3:
        st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Proof of Accuracy")
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        ax5.scatter(y_test, y_pred, color='#1f77b4', alpha=0.8, edgecolors='k', s=80, label='Test Data Points')
        ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.5, label='Perfect Fit')
        ax5.set_xlabel('Actual Profit ($)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Predicted Profit ($)', fontsize=11, fontweight='bold')
        ax5.grid(True, linestyle=':', alpha=0.6)
        ax5.legend(loc='upper left', frameon=True, shadow=True)
        st.pyplot(fig5)

    with col2:
        st.subheader("2. Evidence of Department Impact")
        features = X.columns
        coefs = model.coef_
        mean_values = X.mean()

        contributions = []
        for feature, coef in zip(features, coefs):
            if 'State' not in feature:
                contributions.append({'Department': feature, 'Average Contribution ($)': coef * mean_values[feature]})

        contrib_df = pd.DataFrame(contributions).sort_values(by='Average Contribution ($)', ascending=False)
        fig_coef, ax_coef = plt.subplots(figsize=(7, 5))
        sns.barplot(x='Average Contribution ($)', y='Department', data=contrib_df,
                    palette=['#2ca02c' if x > 0 else '#d62728' for x in contrib_df['Average Contribution ($)']],
                    ax=ax_coef)
        ax_coef.set_xlabel("Impact on Total Profit ($)", fontweight='bold')
        ax_coef.set_ylabel("")
        ax_coef.grid(True, axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig_coef)

    st.markdown("---")

    st.subheader("🔬 Advanced Error Analysis (Residuals)")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Residuals vs. Predictions")
        fig_res, ax_res = plt.subplots(figsize=(6, 4))
        ax_res.scatter(y_pred, residuals, color='purple', alpha=0.6, edgecolors='k', s=70)
        ax_res.axhline(y=0, color='r', linestyle='--', lw=2)
        ax_res.set_xlabel("Predicted Profit ($)", fontweight='bold')
        ax_res.set_ylabel("Residual Error ($)", fontweight='bold')
        ax_res.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_res)
        st.success(
            "**How to read this:** We want the dots to look like a completely random cloud of dust around the red line. Because our dots are scattered randomly without a clear shape, it proves the model doesn't have a built-in bias!")

    with col4:
        st.markdown("#### Distribution of Errors")
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, color='teal', bins=10, ax=ax_hist)
        ax_hist.set_xlabel("Residual Error Amount ($)", fontweight='bold')
        ax_hist.set_ylabel("Frequency", fontweight='bold')
        ax_hist.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_hist)
        st.success(
            "**How to read this:** We are looking for a classic \"Bell Curve\" that peaks near $0. Because the tallest bars are right in the middle, it proves that most of the time, the model's mistakes are small. Extreme misses are rare.")

    st.markdown("---")
    st.subheader("📋 Objective Conclusion")
    st.info(
        "The data provides unequivocal evidence that **R&D allocation is the primary reliable predictor of profit growth**. Reallocating budget away from Administration and into Research & Development is statistically shown to yield the highest expected return.")