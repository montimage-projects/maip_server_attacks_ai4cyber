import pandas as pd
import os
import sys
import joblib

def preprocess_csv(input_file: str, output_file: str = None) -> None:
    """Preprocess the input CSV file by dropping specified columns and saving the modified file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str, optional): Path to save the preprocessed CSV file. Defaults to input_file + '_preprocessed.csv'.
    """
    # Define columns to drop
    columns_to_drop = [
        "Flow ID",
        "Src IP",
        "Src Port",
        "Dst IP",
        "Dst Port",
        "Protocol",
        #"CWR Flag Count",  # only drop for SC1.3
        "Label",
        "Timestamp",
        "output"  # Added output for activity classification
    ]

    # Load the CSV file
    data = pd.read_csv(input_file)

    # Drop specified columns
    data_preprocessed = data.drop(columns=columns_to_drop, axis=1, errors='ignore')

    # Set default output file name if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_preprocessed.csv"

    # Save the preprocessed data to a new CSV file
    data_preprocessed.to_csv(output_file, index=False)

def get_expected_input_features(scaler_path):
    """
    Load the scaler and return the expected number of input features.

    :param scaler_path: Path to the scaler joblib file
    :return: Expected number of input features
    """
    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Check if the scaler has the n_features_in_ attribute
    if hasattr(scaler, 'n_features_in_'):
        return scaler.n_features_in_
    else:
        raise ValueError("The loaded scaler does not have the expected attribute 'n_features_in_'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils.py <input_file.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    preprocess_csv(input_file)
    print(f"Preprocessing complete. Saved as '{os.path.splitext(input_file)[0]}_preprocessed.csv'.")