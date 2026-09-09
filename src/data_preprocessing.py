# ==========================================
# DATA PREPROCESSING MODULE
# ==========================================

import pandas as pd
import re
import os


def clean_text(text):
    """
    Clean news text.
    """

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_data(df):
    """
    Preprocess the fake news dataset.
    """

    print("\n" + "=" * 55)
    print("DATA PREPROCESSING")
    print("=" * 55)


    # -----------------------------------
    # 1. ORIGINAL DATASET SHAPE
    # -----------------------------------

    print("\nOriginal dataset shape:", df.shape)


    # -----------------------------------
    # 2. REMOVE DUPLICATES
    # -----------------------------------

    duplicates_before = df.duplicated().sum()

    print("\nDuplicate rows before removal:", duplicates_before)

    df = df.drop_duplicates()

    duplicates_after = df.duplicated().sum()

    print("Duplicate rows after removal:", duplicates_after)


    # -----------------------------------
    # 3. HANDLE MISSING VALUES
    # -----------------------------------

    print("\nMissing values before handling:")
    print(df.isnull().sum())

    # Remove rows where main news text is missing
    df = df.dropna(subset=["text"])

    # Fill missing titles with empty string
    df["title"] = df["title"].fillna("")

    # Fill missing subject with Unknown
    df["subject"] = df["subject"].fillna("Unknown")

    # Fill missing date with Unknown
    df["date"] = df["date"].fillna("Unknown")

    print("\nMissing values after handling:")
    print(df.isnull().sum())


    # -----------------------------------
    # 4. REMOVE EMPTY TEXT RECORDS
    # -----------------------------------

    df = df[
        df["text"]
        .astype(str)
        .str.strip()
        .ne("")
    ]

    print("\nDataset shape after removing empty text:", df.shape)


    # -----------------------------------
    # 5. COMBINE TITLE AND TEXT
    # -----------------------------------

    df["combined_text"] = (
        df["title"].astype(str)
        + " "
        + df["text"].astype(str)
    )

    print("\nTitle and text combined successfully.")


    # -----------------------------------
    # 6. CLEAN COMBINED TEXT
    # -----------------------------------

    print("\nCleaning text...")

    df["cleaned_text"] = df["combined_text"].apply(clean_text)

    print("Text cleaning completed successfully.")


    # -----------------------------------
    # 7. FINAL DATASET
    # -----------------------------------

    print("\nFinal dataset shape:", df.shape)


    print("\nFinal columns:")
    print(df.columns.tolist())


    # -----------------------------------
    # PREPROCESSING COMPLETE
    # -----------------------------------

    print("\n" + "=" * 55)
    print("🎉 DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 55)

    return df


# ==========================================
# TEST DATA PREPROCESSING MODULE
# ==========================================

if __name__ == "__main__":

    from data_ingestion import load_data

    # Load raw data
    data = load_data()

    # Preprocess data
    processed_data = preprocess_data(data)

    # Create processed data directory
    os.makedirs("../data/processed", exist_ok=True)

    # Save processed dataset
    processed_data.to_csv(
        "../data/processed/news_data.csv",
        index=False
    )

    print("\nProcessed dataset saved successfully!")