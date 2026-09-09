# ==========================================
# MODEL TRAINING MODULE
# ==========================================

import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


def train_model(df):
    """
    Train Fake News Detection model using
    TF-IDF + Random Forest.
    """

    print("\n" + "=" * 55)
    print("MODEL TRAINING")
    print("=" * 55)


    # -----------------------------------
    # 1. SELECT FEATURES AND TARGET
    # -----------------------------------

    X = df["cleaned_text"]
    y = df["label"]

    print("\nFeature selected: cleaned_text")
    print("Target selected: label")


    # -----------------------------------
    # 2. TRAIN-TEST SPLIT
    # -----------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])


    # -----------------------------------
    # 3. TF-IDF VECTORIZATION
    # -----------------------------------

    print("\nCreating TF-IDF features...")

    max_features = 5000

    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english"
    )

    X_train_tfidf = tfidf.fit_transform(X_train)

    X_test_tfidf = tfidf.transform(X_test)

    print("TF-IDF vectorization completed.")

    print(
        "Number of features:",
        X_train_tfidf.shape[1]
    )


    # -----------------------------------
    # 4. RANDOM FOREST MODEL
    # -----------------------------------

    print("\nTraining Random Forest model...")

    n_estimators = 100

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_tfidf, y_train)

    print("Random Forest training completed!")


    # -----------------------------------
    # 5. MLFLOW LOGGING
    # -----------------------------------

    print("\nLogging training parameters to MLflow...")

    mlflow.log_param(
        "model_type",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_features",
        max_features
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_metric(
        "training_samples",
        len(X_train)
    )

    mlflow.log_metric(
        "testing_samples",
        len(X_test)
    )

    print("MLflow training parameters logged successfully!")


    # -----------------------------------
    # 6. CREATE MODELS DIRECTORY
    # -----------------------------------

    os.makedirs("models", exist_ok=True)


    # -----------------------------------
    # 7. SAVE MODEL LOCALLY
    # -----------------------------------

    joblib.dump(
        model,
        "models/random_forest_model.pkl"
    )

    print("\nModel saved successfully!")


    # -----------------------------------
    # 8. SAVE TF-IDF VECTORIZER
    # -----------------------------------

    joblib.dump(
        tfidf,
        "models/tfidf_vectorizer.pkl"
    )

    print("TF-IDF vectorizer saved successfully!")


    # -----------------------------------
    # 9. LOG MODEL TO MLFLOW
    # -----------------------------------

    print("\nLogging model to MLflow...")

    mlflow.sklearn.log_model(
        model,
        artifact_path="random_forest_model"
    )

    print("Model logged to MLflow successfully!")


    # -----------------------------------
    # MODEL TRAINING COMPLETE
    # -----------------------------------

    print("\n" + "=" * 55)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 55)


    return (
        model,
        tfidf,
        X_test,
        y_test
    )


# ==========================================
# TEST MODEL TRAINING MODULE
# ==========================================

if __name__ == "__main__":

    from data_ingestion import load_data
    from data_validation import validate_data
    from data_preprocessing import preprocess_data


    # -----------------------------------
    # START MLFLOW RUN
    # -----------------------------------

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        "Fake_News_Detection"
    )


    with mlflow.start_run():

        # Load data
        data = load_data()

        # Validate data
        validate_data(data)

        # Preprocess data
        processed_data = preprocess_data(data)

        # Train model
        model, tfidf, X_test, y_test = train_model(
            processed_data
        )