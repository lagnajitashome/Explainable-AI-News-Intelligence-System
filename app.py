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
# LOAD SAVED MODEL AND VECTORIZER
# -----------------------------------

model = joblib.load("models/random_forest_model.pkl")

tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# -----------------------------------
# APP TITLE
# -----------------------------------

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article or news text below to check whether it is "
    "likely to be REAL or FAKE."
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
# PREDICTION
# -----------------------------------

if st.button("🔍 Analyze News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")

    else:

        # -----------------------------------
        # CONVERT TEXT TO TF-IDF
        # -----------------------------------

        news_tfidf = tfidf.transform([news_text])


        # -----------------------------------
        # MAKE PREDICTION
        # -----------------------------------

        prediction = model.predict(news_tfidf)[0]

        probabilities = model.predict_proba(news_tfidf)[0]

        confidence = max(probabilities) * 100


        # -----------------------------------
        # DISPLAY RESULT
        # -----------------------------------

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 FAKE NEWS DETECTED")
        else:
            st.success("✅ REAL NEWS DETECTED")


        # -----------------------------------
        # CONFIDENCE
        # -----------------------------------

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        # -----------------------------------
        # PROBABILITIES
        # -----------------------------------

        st.subheader("Prediction Probabilities")

        real_probability = probabilities[0] * 100
        fake_probability = probabilities[1] * 100

        st.write(
            f"🟢 Real News Probability: {real_probability:.2f}%"
        )

        st.write(
            f"🔴 Fake News Probability: {fake_probability:.2f}%"
        )


        # ===================================
        # EXPLAINABLE AI
        # ===================================

        st.subheader("🔍 Explainable AI")

        st.write(
            "The following words were important features present "
            "in the news text according to the Random Forest model."
        )


        # Get all TF-IDF feature names
        feature_names = tfidf.get_feature_names_out()

        # Get TF-IDF values for this news text
        input_features = news_tfidf.toarray()[0]

        # Get indices of words present in the input
        non_zero_indices = input_features.nonzero()[0]

        # Get Random Forest feature importance
        feature_importance = model.feature_importances_


        # -----------------------------------
        # CREATE EXPLANATION DATA
        # -----------------------------------

        explanation_data = []

        for index in non_zero_indices:

            word = feature_names[index]

            importance = feature_importance[index]

            tfidf_score = input_features[index]

            influence = importance * tfidf_score

            explanation_data.append({
                "Word": word,
                "Feature Importance": importance,
                "TF-IDF Score": tfidf_score,
                "Influence": influence
            })


        # -----------------------------------
        # CREATE DATAFRAME
        # -----------------------------------

        explanation_df = pd.DataFrame(explanation_data)


        # Sort by influence
        explanation_df = explanation_df.sort_values(
            by="Influence",
            ascending=False
        )


        # Get top 10 influential words
        top_features = explanation_df.head(10)


        # -----------------------------------
        # DISPLAY TOP WORDS
        # -----------------------------------

        st.subheader("🏆 Top 10 Influential Words")

        st.dataframe(
            top_features,
            use_container_width=True
        )


        # -----------------------------------
        # FEATURE INFLUENCE CHART
        # -----------------------------------

        st.subheader("📊 Feature Influence Chart")

        chart_data = top_features.set_index("Word")["Influence"]

        st.bar_chart(chart_data)