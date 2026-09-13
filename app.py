import streamlit as st
import joblib
import pandas as pd


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)


# -----------------------------------
# LOAD MODEL AND TF-IDF VECTORIZER
# -----------------------------------

model = joblib.load(
    "models/random_forest_model.pkl"
)

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# -----------------------------------
# APP TITLE
# -----------------------------------

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article or news text below to check whether "
    "it is likely to be REAL or FAKE."
)


# -----------------------------------
# USER INPUT
# -----------------------------------

news_text = st.text_area(
    "Enter News Text:",
    height=250,
    placeholder="Paste the news article here..."
)


# -----------------------------------
# ANALYZE NEWS
# -----------------------------------

if st.button("🔍 Analyze News"):

    if not news_text.strip():

        st.warning(
            "Please enter some news text."
        )

    else:

        # ===================================
        # TF-IDF TRANSFORMATION
        # ===================================

        news_tfidf = tfidf.transform(
            [news_text]
        )


        # -----------------------------------
        # FEATURE COVERAGE
        # -----------------------------------

        recognized_features = news_tfidf.nnz
        total_features = news_tfidf.shape[1]

        st.caption(
            f"Recognized TF-IDF features: "
            f"{recognized_features} / {total_features}"
        )


        # ===================================
        # MODEL PREDICTION
        # ===================================

        prediction = model.predict(
            news_tfidf
        )[0]

        probabilities = model.predict_proba(
            news_tfidf
        )[0]


        # -----------------------------------
        # CLASS MAPPING
        #
        # 0 = FAKE
        # 1 = REAL
        # -----------------------------------

        fake_probability = (
            probabilities[0] * 100
        )

        real_probability = (
            probabilities[1] * 100
        )

        confidence = max(
            fake_probability,
            real_probability
        )


        # ===================================
        # PREDICTION RESULT
        # ===================================

        st.subheader(
            "Prediction Result"
        )


        if prediction == 0:

            st.error(
                "🚨 FAKE NEWS DETECTED"
            )

        else:

            st.success(
                "✅ REAL NEWS DETECTED"
            )


        # ===================================
        # CONFIDENCE
        # ===================================

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        # ===================================
        # CONFIDENCE WARNING
        # ===================================

        if confidence < 60:

            st.warning(
                "⚠️ Low-confidence prediction. "
                "The model is uncertain about this article."
            )

        elif confidence < 75:

            st.info(
                "ℹ️ Moderate-confidence prediction."
            )

        else:

            st.success(
                "Model has relatively high confidence "
                "in this prediction."
            )


        # ===================================
        # PREDICTION PROBABILITIES
        # ===================================

        st.subheader(
            "Prediction Probabilities"
        )


        st.write(
            f"🟢 Real News Probability: "
            f"{real_probability:.2f}%"
        )

        st.write(
            f"🔴 Fake News Probability: "
            f"{fake_probability:.2f}%"
        )


        # ===================================
        # PROBABILITY BAR
        # ===================================

        probability_df = pd.DataFrame(
            {
                "Probability": [
                    real_probability,
                    fake_probability
                ]
            },
            index=[
                "Real News",
                "Fake News"
            ]
        )

        st.bar_chart(
            probability_df
        )


        # ===================================
        # EXPLAINABLE AI
        # ===================================

        st.subheader(
            "🔍 Explainable AI"
        )

        st.write(
            "These are the most influential TF-IDF "
            "features present in the input according "
            "to the Random Forest model."
        )


        # -----------------------------------
        # TF-IDF FEATURE NAMES
        # -----------------------------------

        feature_names = (
            tfidf.get_feature_names_out()
        )


        # -----------------------------------
        # INPUT TF-IDF VALUES
        # -----------------------------------

        input_features = (
            news_tfidf.toarray()[0]
        )


        # -----------------------------------
        # FEATURES PRESENT IN INPUT
        # -----------------------------------

        non_zero_indices = (
            input_features.nonzero()[0]
        )


        # -----------------------------------
        # RANDOM FOREST IMPORTANCE
        # -----------------------------------

        feature_importance = (
            model.feature_importances_
        )


        # ===================================
        # CREATE EXPLANATION DATA
        # ===================================

        explanation_data = []


        for index in non_zero_indices:

            word = feature_names[index]

            importance = (
                feature_importance[index]
            )

            tfidf_score = (
                input_features[index]
            )

            influence = (
                importance * tfidf_score
            )

            explanation_data.append(
                {
                    "Word": word,
                    "Feature Importance": importance,
                    "TF-IDF Score": tfidf_score,
                    "Influence": influence
                }
            )


        # ===================================
        # EXPLANATION DATAFRAME
        # ===================================

        explanation_df = pd.DataFrame(
            explanation_data
        )


        if explanation_df.empty:

            st.info(
                "No vocabulary features from the "
                "saved TF-IDF vectorizer were found "
                "in this article."
            )

        else:

            explanation_df = (
                explanation_df.sort_values(
                    by="Influence",
                    ascending=False
                )
            )


            # ===================================
            # TOP 10 FEATURES
            # ===================================

            top_features = (
                explanation_df.head(10)
            )


            st.subheader(
                "🏆 Top 10 Influential Words"
            )


            st.dataframe(
                top_features,
                width="stretch"
            )


            # ===================================
            # FEATURE INFLUENCE CHART
            # ===================================

            st.subheader(
                "📊 Feature Influence Chart"
            )


            chart_data = (
                top_features
                .set_index("Word")["Influence"]
            )


            st.bar_chart(
                chart_data
            )