# ==========================================
# MAIN MLOPS PIPELINE WITH MLFLOW
# ==========================================

import mlflow

from data_ingestion import load_data
from data_validation import validate_data
from data_preprocessing import preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model


def run_pipeline():

    print("\n" + "=" * 60)
    print("🚀 STARTING FAKE NEWS DETECTION MLOPS PIPELINE")
    print("=" * 60)


    # =====================================
    # MLFLOW CONFIGURATION
    # =====================================

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        "Fake_News_Detection"
    )


    # =====================================
    # START MLFLOW RUN
    # =====================================

    with mlflow.start_run():

        print("\n📌 MLflow run started!")


        # =====================================
        # STEP 1: DATA INGESTION
        # =====================================

        print("\n📥 STEP 1: DATA INGESTION")

        data = load_data()


        # Log dataset information
        mlflow.log_metric(
            "total_records",
            len(data)
        )

        mlflow.log_metric(
            "total_columns",
            len(data.columns)
        )


        # =====================================
        # STEP 2: DATA VALIDATION
        # =====================================

        print("\n🔍 STEP 2: DATA VALIDATION")

        validate_data(data)


        # =====================================
        # STEP 3: DATA PREPROCESSING
        # =====================================

        print("\n🧹 STEP 3: DATA PREPROCESSING")

        processed_data = preprocess_data(data)


        # Log processed dataset information
        mlflow.log_metric(
            "processed_records",
            len(processed_data)
        )


        # =====================================
        # STEP 4: MODEL TRAINING
        # =====================================

        print("\n🤖 STEP 4: MODEL TRAINING")

        model, tfidf, X_test, y_test = train_model(
            processed_data
        )


        # =====================================
        # LOG MODEL PARAMETERS
        # =====================================

        print("\n📌 Logging model parameters to MLflow...")

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "n_estimators",
            100
        )

        mlflow.log_param(
            "tfidf_max_features",
            5000
        )

        mlflow.log_param(
            "test_size",
            0.2
        )

        mlflow.log_param(
            "random_state",
            42
        )


        # =====================================
        # STEP 5: MODEL EVALUATION
        # =====================================

        print("\n📊 STEP 5: MODEL EVALUATION")

        metrics = evaluate_model(
            model,
            tfidf,
            X_test,
            y_test
        )


        # =====================================
        # LOG FINAL METRICS
        # =====================================

        print("\n📌 Logging final metrics to MLflow...")

        for metric, value in metrics.items():

            mlflow.log_metric(
                metric,
                value
            )


        # =====================================
        # PIPELINE COMPLETED
        # =====================================

        print("\n" + "=" * 60)
        print("🎉 MLOPS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)


        print("\nFINAL MODEL METRICS:")

        for metric, value in metrics.items():

            print(
                f"{metric}: {value:.4f}"
            )


        print("\n📌 All results logged to MLflow!")


    return metrics


# ==========================================
# RUN PIPELINE
# ==========================================

if __name__ == "__main__":

    run_pipeline()