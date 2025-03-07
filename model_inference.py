import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import random
from typing import Union, Any, Dict
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import tensorflow as tf
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from ctgan import CTGAN
from ctgan import load_demo
from tensorflow.keras.utils import to_categorical
from utils import get_expected_input_features

class ModelInference:
    """Generic class for model inference across different ML frameworks"""

    def __init__(self, model_path: str, scaler_path: str = None, encoder_path: str = None):
        """
        Initialize the model inference class

        Args:
            model_path (str): Path to the saved model
            scaler_path (str, optional): Path to the scaler object
            encoder_path (str, optional): Path to the label encoder
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path

        # Load components
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.encoder = self._load_encoder()

    def _load_model(self) -> Any:
        """Load the model based on file extension"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        ext = os.path.splitext(self.model_path)[1].lower()

        try:
            if ext in ['.h5', '.keras']:
                return load_model(self.model_path)

            elif ext == '.json':
                return lgb.Booster(model_file=self.model_path)

            elif ext == '.model':
                #return xgb.Booster(model_file=self.model_path)
                # Use XGBClassifier for loading the model
                model = xgb.XGBClassifier()
                model.load_model(self.model_path)  # Load the model
                return model

            else:
                raise ValueError(f"Unsupported model format: {ext}")

        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def _load_scaler(self) -> Any:
        """Load the scaler if provided"""
        if self.scaler_path and os.path.exists(self.scaler_path):
            return joblib.load(self.scaler_path)
        return None

    def _load_encoder(self) -> Any:
        """Load the label encoder if provided"""
        if self.encoder_path and os.path.exists(self.encoder_path):
            return joblib.load(self.encoder_path)
        return None

    def label_encoding(self, Y_test: pd.Series):
        '''encode the dataset (label-wise).
        '''
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"The label encoder file at {self.encoder_path} does not exist.")

        enc = joblib.load(self.encoder_path)
        Y_test_encoded = enc.transform(Y_test.to_numpy().reshape(-1))
        return Y_test_encoded, enc

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess the input data"""
        # Get the expected number of input features from the scaler
        expected_features = get_expected_input_features(self.scaler_path)

        # Drop non-feature columns and the output/label column
        columns_to_drop = [
            "Flow ID",
            "Src IP",
            "Src Port",
            "Dst IP",
            "Dst Port",
            "Protocol",
            "Label",
            "Timestamp",
            "output"  # Added output for activity classification
        ]

        # Conditionally drop "CWR Flag Count" based on expected features
        if expected_features == 85:
            columns_to_drop.append("CWR Flag Count")  # For SC 1.3, scaler expects 86 features
        elif expected_features == 86:
            pass  # For SC 1.2, scaler expects 85 features

        X = data.drop(columns=columns_to_drop, axis=1, errors='ignore')

        # Convert to numpy array
        X = X.to_numpy()
        Y_test = data.iloc[:, -1]  # Extract the last column of data

        # Only call label_encoding if the encoder is not None
        if self.encoder is not None:
            Y_test, enc = self.label_encoding(Y_test)
        else:
            Y_test_encoded = Y_test

        # Apply scaling if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X, Y_test

    def predict(self, test_data: pd.DataFrame) -> Dict:
        """Make predictions using the loaded model"""
        # Preprocess the data
        X_test_scaled, y_test = self.preprocess_data(test_data)

        # Make predictions based on model type
        model_type = str(type(self.model))
        #print(model_type)

        if 'xgboost' in model_type:
            #dtest = xgb.DMatrix(X_test_scaled)
            #raw_predictions = self.model.predict(dtest)
            raw_predictions = self.model.predict(X_test_scaled)
            y_pred = np.argmax(raw_predictions, axis=1)

            results = {
                'predictions': y_pred.tolist(),
                'probabilities': raw_predictions.tolist()
            }
            return results

        elif 'lightgbm' in model_type:
            raw_predictions = self.model.predict(X_test_scaled)
            y_pred = np.argmax(raw_predictions, axis=1)

            results = {
                'predictions': y_pred.tolist(),
                'probabilities': raw_predictions.tolist()
            }
            return results

        elif 'keras' in model_type or 'tensorflow' in model_type:
            #X_train_scaled, y_train = self.preprocess_data(train_data)
            X_test_scaled, y_test = self.preprocess_data(test_data)
            #y_test, enc = self.label_encoding(y_test)
            raw_predictions = self.model.predict(X_test_scaled)

            # Check if the predictions are binary or multi-class
            if raw_predictions.ndim == 1:  # Binary classification
                y_pred = (raw_predictions > 0.5).astype(int)  # Apply threshold for binary classification
            else:  # Multi-class classification
                y_pred = np.argmax(raw_predictions, axis=1)  # Get class with highest probability

            results = {
                'predictions': y_pred.tolist(),
                'probabilities': raw_predictions.tolist()
            }
            return results

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

def perform_ctgan_attack(model_inference, poisoning_rate: float, number_epochs: int):
    """
    Perform CTGAN poisoning attack by generating synthetic samples and replacing a percentage of the original samples.

    :param model_inference: An instance of the ModelInference class containing the original training data.
    :param poisoning_rate: Percentage of original samples to replace with synthetic samples.
    :param number_epochs: Number of epochs to train the CTGAN model.
    :return: Poisoned X_train and y_train
    """
    # Load the original training data from the model inference instance
    original_data = model_inference.train_data  # Assuming train_data is an attribute of model_inference

    # Automatically identify discrete columns
    discrete_columns = original_data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Initialize the CTGAN model
    ctgan = CTGAN(epochs=number_epochs)

    # Fit the CTGAN model to the original training data
    ctgan.fit(original_data, discrete_columns)

    # Generate synthetic data equal to the length of the original dataset
    synthetic_data = ctgan.sample(len(original_data))

    # Save synthetic samples to a CSV file
    synthetic_data.to_csv('ctgan.csv', index=False)

    # Calculate number of samples to replace based on poisoning rate
    required_samples = int(len(original_data) * (poisoning_rate / 100))

    # Randomly select indices to replace in the original dataset
    to_replace_idx = random.sample(range(len(original_data)), required_samples)

    # Create poisoned datasets
    X_poisoned = original_data.copy()
    y_poisoned = model_inference.train_labels.copy()  # Assuming train_labels is an attribute of model_inference

    # Replace original samples with synthetic samples
    for idx in to_replace_idx:
        X_poisoned.iloc[idx] = synthetic_data.iloc[idx]

    return X_poisoned, y_poisoned

def perform_random_swap_label_attack(X_train, y_train, poisoning_rate):
    """
    Perform Random Swap Label (RSL) poisoning attack

    :param X_train: Original training features
    :param y_train: Original training labels
    :param poisoning_rate: Percentage of labels to swap (0-100)
    :return: Poisoned X_train and y_train
    """
    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()

    # Calculate the number of labels to swap based on the poisoning rate
    flip_amount = int(len(y_train) * (poisoning_rate / 100))

    # Perform the label swapping
    for _ in range(flip_amount):
        # Select two random indices
        idx1, idx2 = random.sample(range(len(y_poisoned)), 2)
        # Swap their labels
        y_poisoned[idx1], y_poisoned[idx2] = y_poisoned[idx2], y_poisoned[idx1]

    return X_poisoned, y_poisoned

def perform_target_label_flip_attack(X_train, y_train, poisoning_rate, target_class):
    """
    Perform Target Label Flip (TLF) poisoning attack.

    :param X_train: Original training features (numpy array or pandas DataFrame)
    :param y_train: Original training labels (numpy array or pandas Series)
    :param poisoning_rate: Percentage of labels to flip (0-100)
    :param target_class: Target class to flip labels to (must match dtype of y_train)
    :return: Poisoned X_train and y_train
    """
    # Ensure y_train is a numpy array for better manipulation
    y_train = np.array(y_train)  # Convert to numpy array for easier indexing
    X_poisoned = np.array(X_train).copy()
    y_poisoned = y_train.copy()

    # Convert target_class to the same type as y_train's elements
    target_class = type(y_train[0])(target_class)

    # Check that target_class exists in y_train
    unique_labels = set(y_train)
    if target_class not in unique_labels:
        raise ValueError(f"Target class '{target_class}' not found in training labels. Available classes: {unique_labels}")

    # Find all indices of labels that are NOT the target class
    indices_to_flip = np.where(y_poisoned != target_class)[0]

    # Determine how many labels to flip
    flip_amount = int(len(y_train) * (poisoning_rate / 100))
    available_flips = len(indices_to_flip)

    if available_flips < flip_amount:
        print(f"Warning: Requested to flip {flip_amount} labels, but only {available_flips} available. Flipping {available_flips} instead.")
        flip_amount = available_flips  # Adjust to the max available

    # Randomly select indices to flip
    selected_indices = np.random.choice(indices_to_flip, flip_amount, replace=False)

    # Flip the selected labels to the target class
    y_poisoned[selected_indices] = target_class

    return X_poisoned, y_poisoned

def generate_poisoned_dataset(X_train, y_train, attack_type, poisoning_rate, target_class=None, ctgan_file=None):
    """
    Generate poisoned dataset based on attack type

    :param X_train: Original training features
    :param y_train: Original training labels
    :param attack_type: Type of attack ('ctgan', 'rsl', or 'tlf')
    :param poisoning_rate: Percentage of poisoning
    :param target_class: Target class for TLF attack
    :param ctgan_file: Path to CTGAN samples file
    :return: Poisoned X_train and y_train
    """
    if attack_type == 'ctgan':
        if ctgan_file is None:
            raise ValueError("CTGAN attack requires ctgan_file parameter")
        return perform_ctgan_attack(X_train, y_train, poisoning_rate, ctgan_file)

    elif attack_type == 'rsl':
        return perform_random_swap_label_attack(X_train, y_train, poisoning_rate)

    elif attack_type == 'tlf':
        if target_class is None:
            raise ValueError("TLF attack requires target_class parameter")
        return perform_target_label_flip_attack(X_train, y_train, poisoning_rate, target_class)

    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

def retrain_model(model_inference, poisoned_data_path):
    """
    Retrain the model using the poisoned dataset and evaluate its performance on the test dataset.

    :param model_inference: An instance of the ModelInference class
    :param poisoned_data_path: Path to the poisoned training dataset (CSV file)
    :return: Accuracy and confusion matrix of the retrained model
    """
    # Load the poisoned training data
    poisoned_data = pd.read_csv(poisoned_data_path)
    X_train_scaled, y_train = model_inference.preprocess_data(poisoned_data)

    if not os.path.exists(model_inference.model_path):
        raise FileNotFoundError(f"Model file not found: {model_inference.model_path}")

    ext = os.path.splitext(model_inference.model_path)[1].lower()

    try:
        if ext in ['.h5', '.keras']:
            # Determine the number of output units in the model
            num_output_units = model_inference.model.output_shape[1]

            # Check if the model's ID starts with "ac-"
            model_directory = os.path.dirname(model_inference.model_path)
            model_id = os.path.basename(model_directory)

            # TODO: better approach ?
            if model_id.startswith("ac-"):
                # Convert y_train to one-hot encoding
                y_train = to_categorical(y_train, num_classes=num_output_units)

            # Fit the model on the poisoned training data
            model_inference.model.fit(X_train_scaled, y_train)

        elif ext in ['.model', '.json']:  # Handle XGBoost model formats
            # Retrain the model
            model_inference.model.fit(X_train_scaled, y_train)

        else:
            raise ValueError(f"Unsupported model format: {ext}")

    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generic Model Inference with Poisoning Attacks')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')

    # Data arguments group
    parser.add_argument('--test_data', type=str, help='Path to test data (CSV)')
    parser.add_argument('--train_data', type=str, help='Path to training data (CSV)')

    # Optional arguments
    parser.add_argument('--scaler_path', type=str, help='Path to scaler object (optional)')
    parser.add_argument('--encoder_path', type=str, help='Path to label encoder (optional)')

    # Poisoning attack arguments
    parser.add_argument('--attack_type', type=str, choices=['ctgan', 'rsl', 'tlf'],
                       help='Type of poisoning attack to perform')
    parser.add_argument('--poisoning_rate', type=float,
                       help='Percentage of samples to poison (0-100)')
    parser.add_argument('--target_class', type=str,
                       help='Target class for TLF attack')
    parser.add_argument('--ctgan_file', type=str,
                       help='Path to CTGAN generated samples (required for CTGAN attack)')
    parser.add_argument('--output_file', type=str,
                       help='Path to save poisoned dataset')

    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()

    try:
        # Load data
        train_data_path = args.train_data
        train_data = pd.read_csv(train_data_path)
        test_data_path = args.test_data
        test_data = pd.read_csv(test_data_path)

        # If poisoning attack is requested
        if args.attack_type:
            if not args.poisoning_rate:
                raise ValueError("Poisoning rate must be specified when performing an attack")

            # Prepare data for poisoning
            label_column = train_data.columns[-1]
            X = train_data.drop([label_column], axis=1, errors='ignore').values
            y = train_data[label_column].values

            # Generate poisoned dataset
            X_poisoned, y_poisoned = generate_poisoned_dataset(
                X_train=X,
                y_train=y,
                attack_type=args.attack_type,
                poisoning_rate=args.poisoning_rate,
                target_class=args.target_class,
                ctgan_file=args.ctgan_file
            )

            # Create poisoned DataFrame
            poisoned_data = pd.DataFrame(X_poisoned, columns=train_data.drop([label_column], axis=1).columns)
            poisoned_data[label_column] = y_poisoned

            # Save poisoned dataset if output path is provided
            if args.output_file:
                poisoned_data.to_csv(args.output_file, index=False)
                print(f"Poisoned dataset saved to: {args.output_file}")

            # Print statistics about the poisoning
            print("\nPoisoning Attack Statistics:")
            print(f"Attack Type: {args.attack_type}")
            print(f"Original samples: {len(train_data)}")
            print(f"Poisoned samples: {len(poisoned_data)}")
            print("\nLabel Distribution Before Attack:")
            print(train_data[label_column].value_counts())
            print("\nLabel Distribution After Attack:")
            print(poisoned_data[label_column].value_counts())

        # Initialize model inference
        model_inference = ModelInference(
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            encoder_path=args.encoder_path
        )

        if args.attack_type:
            retrain_model(model_inference, args.output_file)

        #X_train_scaled, y_train = model_inference.preprocess_data(poisoned_data)
        X_test_scaled, y_test = model_inference.preprocess_data(test_data)

        #model_inference.model.fit(X_train_scaled, y_train)

        retrain_model(model_inference, args.output_file)

        # Get predictions
        results = model_inference.predict(test_data)
        predictions = np.array(results['predictions'])

        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        print(f'Accuracy: {accuracy:.4f}')
        print('Confusion Matrix:')
        print(cm)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()