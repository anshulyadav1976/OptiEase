import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import sys
import os

# Add the root directory to the path to import from utils and components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.api_settings import render_api_settings
from utils.api import get_explanation

# Configure page
st.set_page_config(page_title="Regression Analysis", page_icon="📈", layout="wide")
st.title("📈 Regression Analysis")
st.markdown("""
This tool helps you analyze relationships between variables using regression analysis.
Upload your data, select variables, and get insights with natural language explanations.
""")

# Get API settings from the component
api_settings = render_api_settings()

# File upload section
st.sidebar.header("📤 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls"],
    help="Your data remains private and is only processed locally"
)

# Main analysis workflow
if uploaded_file is not None:
    try:
        # Load data based on file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"Data loaded successfully! Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        # Privacy notice
        st.info(
            "🔒 Your data is processed locally and is not stored or shared with third parties. Only the analysis context is sent to the AI provider when generating explanations.")

        # Data preview
        with st.expander("Data Preview", expanded=True):
            st.dataframe(df.head())
            st.write("Data Types:")
            st.write(df.dtypes)

        # Variable selection
        st.header("Variable Selection")

        col1, col2 = st.columns(2)
        with col1:
            # Filter for numeric columns for regression
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

            dependent_var = st.selectbox("Select Dependent Variable (Y)", numeric_cols)

            # Multi-select for independent variables
            independent_vars = st.multiselect(
                "Select Independent Variables (X)",
                [col for col in numeric_cols if col != dependent_var],
                max_selections=5
            )

        with col2:
            # Reference variable for logistic regression
            use_reference = st.checkbox("Use Reference Variable (for Logistic Regression)")
            reference_var = None

            if use_reference:
                # noinspection PyArgumentList
                reference_var = st.selectbox("Select Reference Variable", categorical_cols if categorical_cols else [
                    "No categorical variables available"])
                if reference_var != "No categorical variables available":
                    reference_categories = df[reference_var].unique()
                    reference_category = st.selectbox("Select Reference Category", reference_categories)

                    # Convert to binary for logistic regression
                    df['binary_target'] = (df[reference_var] == reference_category).astype(int)
                    st.info(
                        f"Created binary target with {df['binary_target'].sum()} positive cases out of {len(df)} records")

        # Run analysis button
        if st.button("Run Analysis", type="primary") and len(independent_vars) > 0:
            st.header("Analysis Results")

            # Determine regression type
            if use_reference and reference_var != "No categorical variables available":
                regression_type = "Logistic Regression"
                y = df['binary_target']
            elif len(independent_vars) == 1:
                regression_type = "Simple Linear Regression"
                y = df[dependent_var]
            else:
                regression_type = "Multiple Linear Regression"
                y = df[dependent_var]

            X = df[independent_vars]

            # Basic statistics
            with st.expander("Basic Statistics", expanded=True):
                st.subheader("Descriptive Statistics")
                desc_stats = df[independent_vars + [dependent_var]].describe().transpose()
                st.dataframe(desc_stats)

                # Only send summary statistics, not raw data
                explanation = get_explanation(desc_stats.to_string(), "descriptive statistics", api_settings)
                st.markdown("### Explanation")
                st.write(explanation)

            # Regression analysis
            with st.expander("Regression Analysis", expanded=True):
                st.subheader(f"{regression_type} Results")

                # Add constant for statsmodels
                X_with_const = sm.add_constant(X)

                if regression_type == "Logistic Regression":
                    # Logistic regression with statsmodels
                    model = sm.Logit(y, X_with_const).fit(disp=0)
                    st.text(model.summary().as_text())

                    # Coefficients table
                    coef_df = pd.DataFrame({
                        'Variable': ['Intercept'] + independent_vars,
                        'Coefficient': model.params,
                        'Std Error': model.bse,
                        'z-value': model.tvalues,
                        'p-value': model.pvalues
                    })
                    st.dataframe(coef_df)

                    # Equation
                    eq = f"log(p/(1-p)) = {model.params[0]:.4f}"
                    for i, var in enumerate(independent_vars):
                        eq += f" + {model.params[i + 1]:.4f} × {var}"
                    st.markdown(f"### Logistic Regression Equation")
                    st.latex(eq)

                    # Model performance
                    st.markdown("### Model Performance")
                    st.write(f"Pseudo R-squared: {model.prsquared:.4f}")
                    st.write(f"Log-Likelihood: {model.llf:.4f}")
                    st.write(f"AIC: {model.aic:.4f}")
                else:
                    # Linear regression with statsmodels
                    model = sm.OLS(y, X_with_const).fit()
                    st.text(model.summary().as_text())

                    # Coefficients table
                    coef_df = pd.DataFrame({
                        'Variable': ['Intercept'] + independent_vars,
                        'Coefficient': model.params,
                        'Std Error': model.bse,
                        't-value': model.tvalues,
                        'p-value': model.pvalues
                    })
                    st.dataframe(coef_df)

                    # Equation
                    eq = f"{dependent_var} = {model.params[0]:.4f}"
                    for i, var in enumerate(independent_vars):
                        eq += f" + {model.params[i + 1]:.4f} × {var}"
                    st.markdown(f"### Regression Equation")
                    st.latex(eq)

                    # Model performance
                    st.markdown("### Model Performance")
                    st.write(f"R-squared: {model.rsquared:.4f}")
                    st.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
                    st.write(f"F-statistic: {model.fvalue:.4f} (p-value: {model.f_pvalue:.4f})")
                    st.write(f"Mean squared error: {mean_squared_error(y, model.predict(X_with_const)):.4f}")

                # Get explanation - sending only model summary, not raw data
                model_summary = model.summary().as_text()
                explanation = get_explanation(model_summary, regression_type, api_settings)
                st.markdown("### Explanation")
                st.write(explanation)

            # Visualizations
            with st.expander("Visualizations", expanded=True):
                if regression_type != "Logistic Regression":
                    # Actual vs Predicted
                    y_pred = model.predict(X_with_const)

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                    # Actual vs Predicted plot
                    ax1.scatter(y, y_pred, alpha=0.5)
                    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                    ax1.set_xlabel("Actual Values")
                    ax1.set_ylabel("Predicted Values")
                    ax1.set_title("Actual vs Predicted Values")

                    # Residual plot
                    residuals = y - y_pred
                    ax2.scatter(y_pred, residuals, alpha=0.5)
                    ax2.axhline(y=0, color='r', linestyle='--')
                    ax2.set_xlabel("Predicted Values")
                    ax2.set_ylabel("Residuals")
                    ax2.set_title("Residual Plot")

                    st.pyplot(fig)

                    # Correlation heatmap
                    st.subheader("Correlation Heatmap")
                    corr_df = df[independent_vars + [dependent_var]].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    # ROC Curve for logistic regression
                    from sklearn.metrics import roc_curve, auc

                    y_pred_proba = model.predict(X_with_const)
                    fpr, tpr, _ = roc_curve(y, y_pred_proba)
                    roc_auc = auc(fpr, tpr)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                    ax.legend(loc='lower right')
                    st.pyplot(fig)

                # Get visualization explanation - description only, not sending raw data
                viz_explanation_prompt = f"""
                For a {regression_type} with the following characteristics:
                - Dependent variable: {dependent_var}
                - Independent variables: {', '.join(independent_vars)}
                - R-squared: {model.rsquared if hasattr(model, 'rsquared') else model.prsquared:.4f}

                Explain what to look for in:
                1. Actual vs Predicted plot
                2. Residual plot
                3. Correlation heatmap/ROC curve
                """
                viz_explanation = get_explanation(viz_explanation_prompt, "data visualization", api_settings)
                st.markdown("### Visualization Explanation")
                st.write(viz_explanation)

            # Overall analysis summary - sending only aggregated statistics, not raw data
            st.header("Summary Analysis")
            summary_prompt = f"""
            Based on the {regression_type} analysis with {dependent_var} as the target variable 
            and {', '.join(independent_vars)} as predictors:

            Key statistics:
            - R-squared: {model.rsquared if hasattr(model, 'rsquared') else model.prsquared:.4f}
            - {"F-statistic: " + str(model.fvalue) if hasattr(model, 'fvalue') else ""}
            - {"Significant variables: " + ", ".join([var for i, var in enumerate(independent_vars) if model.pvalues[i + 1] < 0.05])}

            Provide a business-friendly summary of what this analysis means.
            What are the key insights? What actions might a business owner take based on this?
            """
            summary = get_explanation(summary_prompt, "overall analysis", api_settings)
            st.write(summary)

            # Chat interface
            st.header("Ask Questions About Your Analysis")
            user_question = st.text_input("Ask a question about your analysis:")

            if user_question:
                # Send only analysis context, not raw data
                analysis_context = f"""
                Analysis type: {regression_type}
                Dependent variable: {dependent_var}
                Independent variables: {', '.join(independent_vars)}
                Key statistics:
                - R-squared: {model.rsquared if hasattr(model, 'rsquared') else model.prsquared:.4f}
                - {"Significant variables (p<0.05): " + ", ".join([var for i, var in enumerate(independent_vars) if model.pvalues[i + 1] < 0.05])}

                Question: {user_question}
                """
                answer = get_explanation(analysis_context, "question", api_settings)
                st.write("### Answer")
                st.write(answer)

    except Exception as e:
        st.error(f"Error analyzing data: {str(e)}")
        st.write("Please check your data format and variable selections.")
else:
    # Show instructions when no data is uploaded
    st.info("👈 Please upload your data file to begin the analysis")

    # Example usage instructions
    st.markdown("""
    ### How to Use This Tool
    1. Upload a CSV or Excel file containing your data
    2. Select your dependent variable (what you want to predict)
    3. Choose one or more independent variables (predictors)
    4. Optionally, select a reference variable for logistic regression
    5. Click "Run Analysis" to generate insights

    ### Sample Data Format
    Your data should contain numerical columns for regression analysis. For example:

    | sales | advertising | price | competitors |
    |-------|------------|-------|------------|
    | 120   | 15         | 29.99 | 3          |
    | 135   | 20         | 24.99 | 2          |
    | ...   | ...        | ...   | ...        |
    """)