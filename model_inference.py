import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import joblib
import random
from typing import Union, Any, Dict
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import tensorflow as tf
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

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

            elif ext == '.pb':
                return tf.saved_model.load(self.model_path)

            elif ext in ['.pkl', '.joblib']:
                return joblib.load(self.model_path)

            elif ext == '.json':
                return lgb.Booster(model_file=self.model_path)

            elif ext == '.model':
                return xgb.Booster(model_file=self.model_path)

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
        # Drop non-feature columns and the output/label column
        X = data.drop([
            "Flow ID",
            "Src IP",
            "Src Port",
            "Dst IP",
            "Dst Port",
            "Protocol",
            "CWR Flag Count", # drop for SC1.3
            "Label",
            "Timestamp",
            "output"  # Added output to the list of columns to drop
        ], axis=1, errors='ignore')

        # Convert to numpy array
        X = X.to_numpy()
        Y_test = data.iloc[:, -1]  # Extract the last column of data
        Y_test, enc = self.label_encoding(Y_test)

        # Apply scaling if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X, Y_test

    def predict(self, data: pd.DataFrame) -> Dict:
        """Make predictions using the loaded model"""
        # Preprocess the data
        X_processed = self.preprocess_data(data)

        # Make predictions based on model type
        model_type = str(type(self.model))
        print(model_type)

        if 'xgboost' in model_type:
            dtest = xgb.DMatrix(X_processed)
            # Get raw predictions (output_margin=True to match training)
            raw_predictions = self.model.predict(dtest, output_margin=True)
            y_pred = (raw_predictions > 0.5)  # Match the threshold used in training

            results = {
                'predictions': y_pred.tolist(),
                'raw_predictions': raw_predictions.tolist()
            }
            return results

        elif 'lightgbm' in model_type:
            # For LightGBM Booster
            raw_predictions = self.model.predict(X_processed)

            # Handle predictions directly as class labels
            if isinstance(raw_predictions, np.ndarray):
                if raw_predictions.dtype == np.float64:
                    y_pred = raw_predictions.round().astype(int)  # Round to nearest integer
                else:
                    y_pred = raw_predictions
            else:
                y_pred = np.array(raw_predictions)

            results = {
                'predictions': y_pred.tolist()
            }
            return results

        elif 'keras' in model_type or 'tensorflow' in model_type:
            #X_train_scaled, y_train = self.preprocess_data(train_data)
            X_test_scaled, y_test = self.preprocess_data(data)
            #y_test, enc = self.label_encoding(y_test)
            raw_predictions = self.model.predict(X_test_scaled)

            # Check if the predictions are binary or multi-class
            if raw_predictions.ndim == 1:  # Binary classification
                y_pred = (raw_predictions > 0.5).astype(int)  # Apply threshold for binary classification
            else:  # Multi-class classification
                y_pred = np.argmax(raw_predictions, axis=1)  # Get class with highest probability

            results = {
                'predictions': y_pred.tolist(),
                'probabilities': raw_predictions.tolist()  # Keep the raw probabilities
            }
            return results

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

def perform_ctgan_attack(X_train, y_train, poisoning_rate, ctgan_file):
    """
    Perform CTGAN poisoning attack

    :param X_train: Original training features
    :param y_train: Original training labels
    :param poisoning_rate: Percentage of poisoned samples to add
    :param ctgan_file: Path to CTGAN generated samples
    :return: Poisoned X_train and y_train
    """
    # Load CTGAN samples
    ctgan_data = pd.read_csv(ctgan_file, delimiter=",")
    X_ctgan = ctgan_data.iloc[:, :-1].values.tolist()
    y_ctgan = ctgan_data.iloc[:, -1].values.tolist()

    # Calculate number of samples to add
    required_samples = int(len(X_train) * poisoning_rate * 0.01)

    # Randomly select and add CTGAN samples
    to_copy_idx = random.sample(range(len(X_ctgan)), required_samples)
    X_poisoned = X_train + [X_ctgan[i] for i in to_copy_idx]
    y_poisoned = y_train + [y_ctgan[i] for i in to_copy_idx]

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

def retrain_model(model_inference, poisoned_data_path, test_data_path):
    """
    Retrain the model using the poisoned dataset and evaluate its performance on the test dataset.

    :param model_inference: An instance of the ModelInference class
    :param poisoned_data_path: Path to the poisoned training dataset (CSV file)
    :param test_data_path: Path to the original testing dataset (CSV file)
    :return: Accuracy and confusion matrix of the retrained model
    """
    # Load the poisoned training data
    poisoned_data = pd.read_csv(poisoned_data_path)
    X_train_scaled, y_train = model_inference.preprocess_data(poisoned_data)

    # Load the test data
    test_data = pd.read_csv(test_data_path)
    X_test_scaled, y_test = model_inference.preprocess_data(test_data)

    # Fit the model on the poisoned training data
    model_inference.model.fit(X_train_scaled, y_train)

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
            retrain_model(model_inference, args.output_file, args.test_data)

        # Make predictions and get confusion matrix if output column exists
        if 'output' in test_data.columns:
            # Get predictions
            results = model_inference.predict(test_data)
            predictions = np.array(results['predictions'])

            # Convert true labels to one-hot encoding (matching training format)
            true_labels = test_data['output'].values
            prep_outputs = [[1,0,0], [0,1,0], [0,0,1]]
            y_true = np.array([prep_outputs[label - 1] for label in true_labels])

            # Calculate accuracy based on model type
            if hasattr(predictions, 'argmax'):
                accuracy = accuracy_score(y_true.argmax(axis=1) + 1,
                                       predictions.argmax(axis=1) + 1)
                cm = confusion_matrix(y_true.argmax(axis=1) + 1,
                                    predictions.argmax(axis=1) + 1)
            else:
                accuracy = accuracy_score(true_labels, predictions)
                cm = confusion_matrix(true_labels, predictions)

            unique_labels = sorted(test_data['output'].unique())

            # Create visualization
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=unique_labels,
                       yticklabels=unique_labels)
            plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=45)
            plt.tight_layout()

            # Save confusion matrix with attack type in filename if applicable
            cm_filename = f'confusion_matrix_{args.attack_type}_{args.poisoning_rate}.png' if args.attack_type else 'confusion_matrix.png'
            plt.savefig(cm_filename)
            plt.close()

            results = {
                'confusion_matrix': cm.tolist(),
                'accuracy': float(accuracy)
            }

            print(json.dumps(results, indent=2))
        else:
            #X_train_scaled, y_train = model_inference.preprocess_data(poisoned_data)
            X_test_scaled, y_test = model_inference.preprocess_data(test_data)

            #model_inference.model.fit(X_train_scaled, y_train)

            retrain_model(model_inference, args.output_file, test_data_path)

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