import pandas as pd


def validate_data(df):
    """
    Validate the dataset before preprocessing
    and model training.
    """

    print("\n" + "=" * 55)
    print("DATA VALIDATION")
    print("=" * 55)


    # -----------------------------------
    # 0. CHECK IF DATASET IS EMPTY
    # -----------------------------------

    if df.empty:
        raise ValueError("Dataset is empty!")

    print("\n✅ Dataset is not empty")


    # -----------------------------------
    # 1. CHECK REQUIRED COLUMNS
    # -----------------------------------

    required_columns = [
        "title",
        "text",
        "subject",
        "date",
        "label"
    ]

    missing_columns = []

    for column in required_columns:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("✅ All required columns are present")


    # -----------------------------------
    # 2. CHECK DATASET SHAPE
    # -----------------------------------

    print("\nDataset Shape:", df.shape)


    # -----------------------------------
    # 3. CHECK MISSING VALUES
    # -----------------------------------

    missing_values = df.isnull().sum()

    total_missing = missing_values.sum()

    print("\nMissing Values:")
    print(missing_values)

    print("\nTotal Missing Values:", total_missing)

    if total_missing > 0:
        print("⚠️ Missing values found!")
    else:
        print("✅ No missing values found")


    # -----------------------------------
    # 4. CHECK DUPLICATE ROWS
    # -----------------------------------

    duplicate_rows = df.duplicated().sum()

    print("\nDuplicate Rows:", duplicate_rows)

    if duplicate_rows > 0:
        print("⚠️ Duplicate rows found!")
    else:
        print("✅ No duplicate rows found")


    # -----------------------------------
    # 5. CHECK DATA TYPES
    # -----------------------------------

    print("\nData Types:")
    print(df.dtypes)


    # -----------------------------------
    # 6. CHECK LABEL VALUES
    # -----------------------------------

    valid_labels = {0, 1}

    dataset_labels = set(df["label"].unique())

    print("\nLabels Found:", dataset_labels)

    if dataset_labels.issubset(valid_labels):
        print("✅ All labels are valid")
    else:
        raise ValueError(
            f"❌ Invalid labels found: {dataset_labels}"
        )


    # -----------------------------------
    # 7. CHECK LABEL DISTRIBUTION
    # -----------------------------------

    print("\nLabel Distribution:")
    print(df["label"].value_counts())


    # -----------------------------------
    # 8. CHECK EMPTY TEXT RECORDS
    # -----------------------------------

    empty_text = (
        df["text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print("\nEmpty Text Records:", empty_text)

    if empty_text > 0:
        print("⚠️ Empty text records found!")
    else:
        print("✅ No empty text records found")


    # -----------------------------------
    # FINAL VALIDATION SUMMARY
    # -----------------------------------

    print("\n" + "=" * 55)
    print("       DATA VALIDATION SUMMARY")
    print("=" * 55)

    print("\nDataset Shape:", df.shape)

    print("\nRequired Columns:")
    print("✅ All required columns present")

    print("\nTotal Missing Values:", total_missing)

    print("Duplicate Rows:", duplicate_rows)

    print("\nLabels Found:", dataset_labels)

    print("\n🎉 DATA VALIDATION COMPLETED SUCCESSFULLY!")

    print("=" * 55)

    return True


# -----------------------------------
# TEST DATA VALIDATION MODULE
# -----------------------------------

if __name__ == "__main__":

    from data_ingestion import load_data

    data = load_data()

    validate_data(data)