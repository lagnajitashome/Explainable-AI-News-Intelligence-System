import streamlit as st
import joblib
import pandas as pd
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="News Intelligence System",
    page_icon="📰",
    layout="centered"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # Random Forest
    model = joblib.load(
        "models/random_forest_model.pkl"
    )

    # TF-IDF
    tfidf = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    # LDA
    lda_model = joblib.load(
        "models/lda_topic_model.pkl"
    )

    # Topic vectorizer
    topic_vectorizer = joblib.load(
        "models/topic_vectorizer.pkl"
    )

    # Semantic artifacts
    with open(
        "models/semantic_topic_embeddings.pkl",
        "rb"
    ) as f:

        semantic_artifacts = pickle.load(f)

    topic_descriptions = semantic_artifacts[
        "topic_descriptions"
    ]

    topic_embeddings = semantic_artifacts[
        "topic_embeddings"
    ]

    # Sentence Transformer
    semantic_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return (
        model,
        tfidf,
        lda_model,
        topic_vectorizer,
        topic_descriptions,
        topic_embeddings,
        semantic_model
    )


(
    model,
    tfidf,
    lda_model,
    topic_vectorizer,
    topic_descriptions,
    topic_embeddings,
    semantic_model
) = load_models()


# ============================================================
# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📰 News Intelligence System")

st.caption(
    "AI-powered news classification, explainability, "
    "topic discovery and semantic analysis"
)


# MODEL / VECTORIZER VALIDATION
# ============================================================

if list(model.classes_) != [0, 1]:

    st.error(
        f"❌ Unexpected model classes: {model.classes_}"
    )

    st.stop()


if model.n_features_in_ != len(
    tfidf.get_feature_names_out()
):

    st.error(
        "❌ MODEL / TF-IDF FEATURE MISMATCH"
    )

    st.write(
        "Model expects:",
        model.n_features_in_
    )

    st.write(
        "TF-IDF provides:",
        len(tfidf.get_feature_names_out())
    )

    st.stop()


# Model/vectorizer validation passed internally.
# Diagnostic details are intentionally hidden from the final UI.


# ============================================================
# USER INPUT
# ============================================================

st.subheader("📝 Enter News Article")

news_text = st.text_area(
    "Paste the complete news article:",
    height=300,
    placeholder="Paste the news title + article text here..."
)


# ============================================================
# ANALYZE
# ============================================================

if st.button("🔍 Analyze News", type="primary", use_container_width=True):

    if not news_text.strip():

        st.warning(
            "Please enter some news text."
        )

        st.stop()


    # ========================================================
    # Input diagnostics are intentionally hidden in the final UI.\n\n\n# TF-IDF TRANSFORMATION
    # ========================================================

    news_tfidf = tfidf.transform(
        [news_text]
    )

    recognized_features = news_tfidf.nnz

    total_features = news_tfidf.shape[1]

    st.write(
        "Recognized TF-IDF features:",
        recognized_features
    )

    st.write(
        "Total TF-IDF features:",
        total_features
    )


    if recognized_features == 0:

        st.error(
            "❌ ZERO vocabulary features recognized!"
        )

        st.warning(
            "The model cannot make a meaningful "
            "prediction from this input."
        )

        st.stop()


    # ========================================================
    # RAW MODEL PREDICTION
    # ========================================================

    prediction = model.predict(
        news_tfidf
    )[0]

    probabilities = model.predict_proba(
        news_tfidf
    )[0]


    # ========================================================
    # IMPORTANT:
    # MAP PROBABILITIES USING MODEL CLASSES
    # ========================================================

    class_probabilities = dict(
        zip(
            model.classes_,
            probabilities
        )
    )

    fake_probability = (
        class_probabilities[0] * 100
    )

    real_probability = (
        class_probabilities[1] * 100
    )

    confidence = max(
        fake_probability,
        real_probability
    )


    # ========================================================
    # Raw model diagnostics are intentionally hidden in the final UI.\n\n\n# FINAL PREDICTION
    # ========================================================

    st.subheader(
        "🎯 Prediction"
    )

    if prediction == 0:

        st.error(
            "🚨 FAKE NEWS DETECTED"
        )

    elif prediction == 1:

        st.success(
            "✅ REAL NEWS DETECTED"
        )

    else:

        st.error(
            f"Unexpected prediction: {prediction}"
        )


    st.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )


    # ========================================================
    # PROBABILITY BREAKDOWN
    # ========================================================

    st.subheader(
        "📊 Prediction Probabilities"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Fake News",
            f"{fake_probability:.2f}%"
        )

    with col2:

        st.metric(
            "Real News",
            f"{real_probability:.2f}%"
        )


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


    # ========================================================
    # TOP INFLUENTIAL WORDS
    # ========================================================

    st.subheader(
        "🔍 Explainable AI"
    )

    feature_names = (
        tfidf.get_feature_names_out()
    )

    input_features = (
        news_tfidf.toarray()[0]
    )

    non_zero_indices = (
        input_features.nonzero()[0]
    )

    feature_importance = (
        model.feature_importances_
    )

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


    explanation_df = pd.DataFrame(
        explanation_data
    )


    if not explanation_df.empty:

        explanation_df = (
            explanation_df
            .sort_values(
                by="Influence",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            explanation_df,
            width="stretch"
        )


    # ========================================================
    # LDA TOPIC MODELING
    # ========================================================

    st.subheader(
        "🧠 Topic Discovery"
    )

    topic_features = (
        topic_vectorizer.transform(
            [news_text]
        )
    )

    topic_distribution = (
        lda_model.transform(
            topic_features
        )[0]
    )

    dominant_topic = (
        int(
            topic_distribution.argmax()
        ) + 1
    )

    topic_probability = (
        topic_distribution[
            dominant_topic - 1
        ] * 100
    )

    topic_description = (
        topic_descriptions.get(
            dominant_topic,
            "General News"
        )
    )

    st.write(
        f"Dominant Topic: Topic {dominant_topic}"
    )

    st.info(
        f"📌 Topic Theme: {topic_description}"
    )

    st.metric(
        "LDA Topic Probability",
        f"{topic_probability:.2f}%"
    )


    # ========================================================
    # SEMANTIC ANALYSIS
    # ========================================================

    st.subheader(
        "🔎 Semantic Analysis"
    )

    article_embedding = (
        semantic_model.encode(
            [news_text],
            convert_to_numpy=True
        )
    )

    semantic_similarities = (
        cosine_similarity(
            article_embedding,
            topic_embeddings
        )[0]
    )

    semantic_topic_index = (
        semantic_similarities.argmax()
    )

    semantic_topic = (
        semantic_topic_index + 1
    )

    semantic_similarity = (
        semantic_similarities[
            semantic_topic_index
        ]
    )

    semantic_description = (
        topic_descriptions.get(
            semantic_topic,
            "General News"
        )
    )

    st.write(
        f"Semantic Topic: Topic {semantic_topic}"
    )

    st.info(
        f"🧩 Semantic Theme: {semantic_description}"
    )

    st.metric(
        "Semantic Similarity",
        f"{semantic_similarity:.2%}"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.subheader(
        "📊 News Intelligence Summary"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Classification",
            "FAKE" if prediction == 0 else "REAL"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with col2:

        st.metric(
            "LDA Topic",
            f"Topic {dominant_topic}"
        )

        st.metric(
            "Semantic Topic",
            f"Topic {semantic_topic}"
        )


    st.caption(
        "⚠️ This system provides an ML-based "
        "prediction and should not be treated as "
        "an absolute determination of factual truth."
    )