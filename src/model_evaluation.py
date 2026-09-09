# ==========================================
# MODEL EVALUATION MODULE
# ==========================================

import os
import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model(model, tfidf, X_test, y_test):
    """
    Evaluate the trained Fake News Detection model.
    """

    print("\n" + "=" * 55)
    print("MODEL EVALUATION")
    print("=" * 55)


    # -----------------------------------
    # 1. CONVERT TEST DATA TO TF-IDF
    # -----------------------------------

    print("\nTransforming test data using TF-IDF...")

    X_test_tfidf = tfidf.transform(X_test)

    print("TF-IDF transformation completed!")


    # -----------------------------------
    # 2. MAKE PREDICTIONS
    # -----------------------------------

    print("\nMaking predictions...")

    y_pred = model.predict(X_test_tfidf)

    print("Predictions completed!")


    # -----------------------------------
    # 3. CALCULATE METRICS
    # -----------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # -----------------------------------
    # 4. DISPLAY METRICS
    # -----------------------------------

    print("\n" + "-" * 55)
    print("MODEL PERFORMANCE METRICS")
    print("-" * 55)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


    # -----------------------------------
    # 5. LOG METRICS TO MLFLOW
    # -----------------------------------

    print("\nLogging evaluation metrics to MLflow...")

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    print("Evaluation metrics logged successfully!")


    # -----------------------------------
    # 6. CLASSIFICATION REPORT
    # -----------------------------------

    print("\n" + "-" * 55)
    print("CLASSIFICATION REPORT")
    print("-" * 55)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Real", "Fake"],
        zero_division=0
    )

    print(report)


    # -----------------------------------
    # 7. CONFUSION MATRIX
    # -----------------------------------

    print("\n" + "-" * 55)
    print("CONFUSION MATRIX")
    print("-" * 55)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)


    # -----------------------------------
    # 8. CREATE OUTPUT DIRECTORY
    # -----------------------------------

    os.makedirs(
        "outputs",
        exist_ok=True
    )


    # -----------------------------------
    # 9. SAVE CLASSIFICATION REPORT
    # -----------------------------------

    report_path = "outputs/classification_report.txt"

    with open(
        report_path,
        "w"
    ) as file:

        file.write(report)


    # Log report as MLflow artifact
    mlflow.log_artifact(
        report_path
    )


    # -----------------------------------
    # 10. CREATE CONFUSION MATRIX PLOT
    # -----------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "Fake"]
    )

    display.plot()

    plt.title(
        "Fake News Detection - Confusion Matrix"
    )


    # -----------------------------------
    # 11. SAVE CONFUSION MATRIX
    # -----------------------------------

    confusion_matrix_path = (
        "outputs/confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


    # Log confusion matrix as MLflow artifact
    mlflow.log_artifact(
        confusion_matrix_path
    )

    print(
        "\nClassification report and confusion matrix "
        "logged to MLflow!"
    )


    # -----------------------------------
    # MODEL EVALUATION COMPLETE
    # -----------------------------------

    print("\n" + "=" * 55)
    print("🎉 MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 55)


    # -----------------------------------
    # RETURN METRICS
    # -----------------------------------

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics


# ==========================================
# TEST MODEL EVALUATION MODULE
# ==========================================

if __name__ == "__main__":

    import mlflow

    from data_ingestion import load_data
    from data_validation import validate_data
    from data_preprocessing import preprocess_data
    from model_training import train_model


    # -----------------------------------
    # MLFLOW CONFIGURATION
    # -----------------------------------

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        "Fake_News_Detection"
    )


    # -----------------------------------
    # START MLFLOW RUN
    # -----------------------------------

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

        # Evaluate model
        metrics = evaluate_model(
            model,
            tfidf,
            X_test,
            y_test
        )