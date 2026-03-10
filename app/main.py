import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


st.title("AI Data Analyst Agent")

uploaded_file = st.file_uploader("Upload your dataset (CSV)")


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Basic Statistics")
    st.write(df.describe())

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=['float64','int64'])

    if not numeric_df.empty:

        corr = numeric_df.corr()

        fig, ax = plt.subplots()

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

    # -----------------------------
    # Histograms
    # -----------------------------

    st.subheader("Histogram of Numeric Features")

    numeric_df.hist(figsize=(10,8))

    st.pyplot(plt)

    # -----------------------------
    # Machine Learning Section
    # -----------------------------

    st.subheader("Automatic Machine Learning Model")

    target = st.selectbox(
        "Select target variable",
        df.columns
    )

    if target:

        X = df.drop(columns=[target])
        y = df[target]

        X = X.select_dtypes(include=['float64','int64'])

        if not X.empty:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            model = LinearRegression()

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            score = r2_score(y_test, predictions)

            st.write("Model R2 Score:", score)

            # -----------------------------
            # Feature Importance
            # -----------------------------

            st.subheader("Feature Importance")

            importance = model.coef_

            feature_importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": importance
            })

            feature_importance = feature_importance.sort_values(
                by="Importance",
                ascending=False
            )

            st.dataframe(feature_importance)

            fig2, ax2 = plt.subplots()

            sns.barplot(
                x="Importance",
                y="Feature",
                data=feature_importance,
                ax=ax2
            )

            st.pyplot(fig2)

            # -----------------------------
            # AI Insight Generator
            # -----------------------------

            st.subheader("AI Insight Generator")

            if st.button("Generate Insight"):

                corr_matrix = numeric_df.corr()

                corr_pairs = corr_matrix.unstack().sort_values(ascending=False)

                corr_pairs = corr_pairs[corr_pairs != 1]

                top_pair = corr_pairs.index[0]

                top_value = corr_pairs.iloc[0]

                insight = f"""
Key Insight:

The strongest relationship in the dataset is between **{top_pair[0]}** and **{top_pair[1]}**
with a correlation of **{round(top_value,2)}**.

Model Interpretation:

The regression model achieved an **R² score of {round(score,2)}**, indicating
the predictive capability of the model.

Recommendation:

Features highly correlated with the target variable may have significant
influence on predictions and should be considered important predictors.
"""

                st.write(insight)
