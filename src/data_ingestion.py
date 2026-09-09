import pandas as pd
import os


def load_data():

    # Get the project root directory
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    # Create paths for datasets
    fake_path = os.path.join(
        project_root,
        "data",
        "Fake.csv"
    )

    true_path = os.path.join(
        project_root,
        "data",
        "True.csv"
    )

    # Load datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add labels
    fake_df["label"] = 1
    true_df["label"] = 0

    # Combine datasets
    df = pd.concat(
        [fake_df, true_df],
        ignore_index=True
    )

    return df


if __name__ == "__main__":

    data = load_data()

    print("Data loaded successfully!")

    print("Dataset shape:", data.shape)

    print("\nFirst 5 rows:")

    print(data.head())