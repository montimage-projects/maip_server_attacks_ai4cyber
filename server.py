from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from model_inference import ModelInference, generate_poisoned_dataset, retrain_model
import uuid
from fastapi.responses import JSONResponse, FileResponse, Response
import shutil
from typing import List

app = FastAPI()

# Paths
UPLOAD_FOLDER = "uploads"
MODEL_PATH = os.path.join(UPLOAD_FOLDER, "model.h5")
TRAIN_DATA_PATH = os.path.join(UPLOAD_FOLDER, "train.csv")
TEST_DATA_PATH = os.path.join(UPLOAD_FOLDER, "test.csv")
SCALER_PATH = os.path.join(UPLOAD_FOLDER, "scaler.joblib")
ENCODER_PATH = os.path.join(UPLOAD_FOLDER, "encoder.joblib")

# Store model instance globally
model_inference = None

@app.post("/upload", summary="Upload model, training/testing dataset, scaler, and encoder.")
async def upload_files(
    model: UploadFile = File(...),
    train_data: UploadFile = File(...),
    test_data: UploadFile = File(...),
    scaler: UploadFile = File(None),
    encoder: UploadFile = File(None),
):
    """Upload model, training/testing data, scaler, and encoder."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Create a unique subfolder for this upload using UUID
    model_id = str(uuid.uuid4())
    model_folder = os.path.join(UPLOAD_FOLDER, model_id)
    os.makedirs(model_folder, exist_ok=True)

    # Update file paths to include the model folder
    model_path = os.path.join(model_folder, model.filename)  # Use the uploaded model's filename
    train_data_path = os.path.join(model_folder, "train.csv")
    test_data_path = os.path.join(model_folder, "test.csv")
    scaler_path = os.path.join(model_folder, "scaler.joblib")
    encoder_path = os.path.join(model_folder, "encoder.joblib")

    # Save files
    with open(model_path, "wb") as f:
        f.write(model.file.read())
    with open(train_data_path, "wb") as f:
        f.write(train_data.file.read())
    with open(test_data_path, "wb") as f:
        f.write(test_data.file.read())

    if scaler:
        with open(scaler_path, "wb") as f:
            f.write(scaler.file.read())
    if encoder:
        with open(encoder_path, "wb") as f:
            f.write(encoder.file.read())

    global model_inference
    model_inference = ModelInference(model_path, scaler_path, encoder_path)

    return {"message": "Files uploaded successfully", "model_id": model_id}


@app.get("/models", summary="List all models.", response_model=dict)
async def list_models():
    """List all models in the uploads directory."""
    uploads_dir = os.path.join(UPLOAD_FOLDER)

    # Check if the uploads directory exists
    if not os.path.exists(uploads_dir):
        raise HTTPException(status_code=404, detail="Uploads directory not found.")

    # List all subdirectories in the uploads directory
    model_folders = [name for name in os.listdir(uploads_dir) if os.path.isdir(os.path.join(uploads_dir, name))]

    return {"models": model_folders}


@app.post("/models/{model_id}/rename", summary="Rename the specified model.")
async def rename_model(model_id: str, new_name: str):
    """
    Rename the model folder from UUID to a meaningful name.
    """
    # Construct the current model folder path
    current_model_folder = os.path.join(UPLOAD_FOLDER, model_id)

    # Check if the current model folder exists
    if not os.path.exists(current_model_folder):
        raise HTTPException(status_code=404, detail="Model folder not found.")

    # Construct the new model folder path
    new_model_folder = os.path.join(UPLOAD_FOLDER, new_name)

    # Check if the new name already exists
    if os.path.exists(new_model_folder):
        raise HTTPException(status_code=400, detail="A model with this name already exists.")

    # Rename the model folder
    shutil.move(current_model_folder, new_model_folder)

    return {"message": f"Model '{model_id}' renamed to '{new_name}' successfully."}


@app.get("/models/{model_id}/train", summary="View or download the training dataset for the specified model.")
async def handle_training_data(
    model_id: str,
    action: str = Query("view", enum=["view", "download"])
):
    """View or download the training dataset for the specified model."""
    model_folder = os.path.join(UPLOAD_FOLDER, model_id)
    train_data_path = os.path.join(model_folder, "train.csv")

    # Check if the training data file exists
    if not os.path.exists(train_data_path):
        raise HTTPException(status_code=404, detail="Training data file not found.")

    if action == "download":
        return FileResponse(train_data_path, media_type='text/csv', filename="train.csv")
    elif action == "view":
        # Read the CSV file and return its content as plain text
        with open(train_data_path, 'r') as file:
            content = file.read()
        return Response(content, media_type='text/csv')  # Serve the CSV content directly
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'view' or 'download'.")


@app.get("/models/{model_id}/test", summary="View or download the testing dataset for the specified model.")
async def handle_testing_data(
    model_id: str,
    action: str = Query("view", enum=["view", "download"])
):
    """View or download the testing dataset for the specified model."""
    model_folder = os.path.join(UPLOAD_FOLDER, model_id)
    test_data_path = os.path.join(model_folder, "test.csv")

    # Check if the testing data file exists
    if not os.path.exists(test_data_path):
        raise HTTPException(status_code=404, detail="Testing data file not found.")

    if action == "download":
        return FileResponse(test_data_path, media_type='text/csv', filename="test.csv")
    elif action == "view":
        # Read the CSV file and return its content as plain text
        with open(test_data_path, 'r') as file:
            content = file.read()
        return Response(content, media_type='text/csv')  # Serve the CSV content directly
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'view' or 'download'.")


@app.get("/evaluate", summary="Evaluate model accuracy & confusion matrix for the specified model.")
async def evaluate_model(model_id: str):
    """Evaluate model accuracy & confusion matrix for the specified model."""
    # Load the specified model based on model_id
    model_folder = os.path.join(UPLOAD_FOLDER, model_id)

    # Check if the model folder exists
    if not os.path.exists(model_folder):
        return {"error": "Model not found."}

    # Load the model from the specified folder
    model_files = [f for f in os.listdir(model_folder) if f.endswith(('.h5', '.keras', '.json', '.model'))]

    if not model_files:
        return {"error": "No model file found in the specified folder."}

    # Assuming we want to load the first model file found
    model_path = os.path.join(model_folder, model_files[0])
    scaler_path = os.path.join(model_folder, "scaler.joblib")
    encoder_path = os.path.join(model_folder, "encoder.joblib")

    # Ensure the test data path is correct
    test_data_path = os.path.join(model_folder, "test.csv")

    global model_inference
    model_inference = ModelInference(model_path, scaler_path, encoder_path)

    # Proceed with evaluation
    test_data = pd.read_csv(test_data_path)
    X_test_scaled, y_test = model_inference.preprocess_data(test_data)

    results = model_inference.predict(test_data)
    predictions = results["predictions"]

    accuracy = accuracy_score(y_test, predictions)
    #accuracy = (predictions == y_test).mean()
    cm = pd.crosstab(y_test, predictions, rownames=["Actual"], colnames=["Predicted"])

    return {
        "model_id": model_id,
        "accuracy": accuracy,
        "confusion_matrix": cm.values.tolist()
    }


@app.post("/attacks/poisoning/ctgan", summary="Apply CTGAN poisoning attack to the training dataset of the specified model.")
async def apply_ctgan_poisoning(
    model_id: str,
    poisoning_rate: str,
    target_class: str = None,
    ctgan_file: str = None
):
    """Apply CTGAN poisoning attack to the training dataset of the specified model."""
    poisoning_rate_float = float(poisoning_rate)
    return await apply_poisoning(model_id, "ctgan", poisoning_rate_float, target_class, ctgan_file)


@app.post("/attacks/poisoning/random-swapping-labels", summary="Apply random swapping labels poisoning attack to the training dataset of the specified model.")
async def apply_random_swapping_labels_poisoning(
    model_id: str,
    poisoning_rate: str
):
    """Apply random swapping labels poisoning attack to the training dataset of the specified model."""
    poisoning_rate_float = float(poisoning_rate)
    return await apply_poisoning(model_id, "rsl", poisoning_rate_float)


@app.post("/attacks/poisoning/target-label-flipping", summary="Apply target label flipping poisoning attack to the training dataset of the specified model.")
async def apply_target_label_flipping_poisoning(
    model_id: str,
    poisoning_rate: str,
    target_class: str = None
):
    """Apply target label flipping poisoning attack to the training dataset of the specified model."""
    poisoning_rate_float = float(poisoning_rate)
    return await apply_poisoning(model_id, "tlf", poisoning_rate_float, target_class)


async def apply_poisoning(model_id: str, attack_type: str, poisoning_rate: float, target_class: str = None, ctgan_file: str = None):
    """Common function to apply poisoning attack to the training dataset of the specified model."""
    # Load the specified model based on model_id
    model_folder = os.path.join(UPLOAD_FOLDER, model_id)

    # Check if the model folder exists
    if not os.path.exists(model_folder):
        return {"error": "Model not found."}

    # Load the training data from the specified model folder
    train_data_path = os.path.join(model_folder, "train.csv")

    # Load the training data
    train_data = pd.read_csv(train_data_path)
    label_column = train_data.columns[-1]
    X_train = train_data.drop([label_column], axis=1, errors='ignore').values
    y_train = train_data[label_column].values

    # Convert poisoning rate from percentage to decimal for calculations
    poisoning_rate_decimal = poisoning_rate / 100.0

    # Create a descriptive filename for the poisoned data for CTGAN
    poisoned_data_filename = "poisoned_data.csv"
    try:
        if attack_type == "ctgan":
            poisoning_rate_int = int(poisoning_rate)
            poisoned_data_filename = f"poisoned_train_ctgan_rate_{poisoning_rate_int}.csv"
        elif attack_type == "rsl":
            poisoning_rate_int = int(poisoning_rate)
            poisoned_data_filename = f"poisoned_train_rsl_rate_{poisoning_rate_int}.csv"
        elif attack_type == "tlf":
            poisoning_rate_int = int(poisoning_rate)
            poisoned_data_filename = f"poisoned_train_tlf_rate_{poisoning_rate_int}_class_{target_class}.csv"
            # Check that target_class exists in y_train
            unique_labels = set(y_train)
            if target_class not in unique_labels:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target class '{target_class}' not found in training labels. Available classes: {unique_labels}"
                )
        else:
            raise ValueError("Invalid attack type.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate poisoned dataset
    X_poisoned, y_poisoned = generate_poisoned_dataset(
        X_train=X_train,
        y_train=y_train,
        attack_type=attack_type,
        poisoning_rate=poisoning_rate,
        target_class=target_class,
        ctgan_file=ctgan_file
    )

    # Create poisoned DataFrame
    poisoned_data = pd.DataFrame(X_poisoned, columns=train_data.drop([label_column], axis=1).columns)
    poisoned_data[label_column] = y_poisoned

    # Save poisoned dataset
    poisoned_data_path = os.path.join(model_folder, poisoned_data_filename)
    poisoned_data.to_csv(poisoned_data_path, index=False)
    print(f"Poisoned dataset saved to: {poisoned_data_path}")

    # Print statistics about the poisoning
    print("\nPoisoning Attack Statistics:")
    print(f"Attack Type: {attack_type}")
    print(f"Original samples: {len(train_data)}")
    print(f"Poisoned samples: {len(poisoned_data)}")

    # Calculate and print label distributions
    original_distribution = train_data[label_column].value_counts()
    poisoned_distribution = poisoned_data[label_column].value_counts()
    print("\nLabel Distribution Before Attack:")
    print(original_distribution)
    print("\nLabel Distribution After Attack:")
    print(poisoned_distribution)

    return {
        "message": f"{attack_type.replace('-', ' ').title()} poisoning attack applied",
        "poisoning_rate": poisoning_rate_int,
        "poisoned_data_path": poisoned_data_path
    }


@app.get("/models/{model_id}/poisoned-datasets", summary="List all poisoned datasets for the specified model.")
async def list_poisoned_datasets(model_id: str):
    """List all poisoned datasets for the specified model."""
    model_folder = os.path.join(UPLOAD_FOLDER, model_id)

    # Check if the model folder exists
    if not os.path.exists(model_folder):
        raise HTTPException(status_code=404, detail="Model not found.")

    # List all files that start with "poisoned_"
    poisoned_datasets = [
        filename for filename in os.listdir(model_folder)
        if filename.startswith("poisoned_train")
    ]

    return {"poisoned_datasets": poisoned_datasets}


@app.get("/models/{model_id}/poisoned-datasets/{dataset_name}", summary="View or download a specific poisoned dataset for the specified model.")
async def get_poisoned_dataset(model_id: str, dataset_name: str, action: str = Query("view", enum=["view", "download"])):
    """View or download a specific poisoned dataset for the specified model."""
    model_folder = os.path.join(UPLOAD_FOLDER, model_id)
    poisoned_data_path = os.path.join(model_folder, dataset_name)

    # Check if the poisoned dataset file exists
    if not os.path.exists(poisoned_data_path):
        raise HTTPException(status_code=404, detail="Poisoned dataset not found.")

    if action == "download":
        return FileResponse(poisoned_data_path, media_type='text/csv', filename=dataset_name)
    elif action == "view":
        # Read the CSV file and return its content as plain text
        with open(poisoned_data_path, 'r') as file:
            content = file.read()
        return Response(content, media_type='text/csv')
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'view' or 'download'.")


@app.post("/impact", summary="Retrain the model on poisoned data & evaluate impact.")
async def impact(model_id: str, poisoned_data_filename: str):
    """Retrain the model on poisoned data & evaluate impact."""
    model_folder = os.path.join(UPLOAD_FOLDER, model_id)

    # Check if the model folder exists
    if not os.path.exists(model_folder):
        return {"error": "Model not found."}

    # Load the model from the specified folder
    model_files = [f for f in os.listdir(model_folder) if f.endswith(('.h5', '.keras', '.json', '.model'))]

    if not model_files:
        return {"error": "No model file found in the specified folder."}

    # Load the model, scaler, and encoder from the specified folder
    model_path = os.path.join(model_folder, model_files[0])
    scaler_path = os.path.join(model_folder, "scaler.joblib")
    encoder_path = os.path.join(model_folder, "encoder.joblib")

    global model_inference
    model_inference = ModelInference(model_path, scaler_path, encoder_path)

    # Load the original train/test data for evaluation
    train_data_path = os.path.join(model_folder, "train.csv")
    if not os.path.exists(train_data_path):
        return {"error": f"Train data file not found: {train_data_path}"}
    test_data_path = os.path.join(model_folder, "test.csv")
    if not os.path.exists(test_data_path):
        return {"error": f"Test data file not found: {test_data_path}"}

    train_data = pd.read_csv(train_data_path)
    X_train_scaled, y_test = model_inference.preprocess_data(train_data)
    test_data = pd.read_csv(test_data_path)
    X_test_scaled, y_test = model_inference.preprocess_data(test_data)

    # Evaluate before retraining
    results = model_inference.predict(test_data)
    #predictions = np.array(results['predictions'])
    #accuracy_before = accuracy_score(y_test, predictions)
    predictions = results["predictions"]
    accuracy_before = (predictions == y_test).mean()
    print(f'Accuracy before: {accuracy_before:.4f}')

    # Load poisoned dataset
    poisoned_data_path = os.path.join(model_folder, poisoned_data_filename)
    if not os.path.exists(poisoned_data_path):
        return {"error": f"Poisoned dataset not found: {poisoned_data_path}"}

    # Retrain model using poisoned training dataset
    retrain_model(model_inference, poisoned_data_path)

    # Evaluate after retraining
    results = model_inference.predict(test_data)
    #predictions = np.array(results['predictions'])
    #accuracy_after = accuracy_score(y_test, predictions)
    predictions = results["predictions"]
    accuracy_after = (predictions == y_test).mean()
    print(f'Accuracy after: {accuracy_after:.4f}')

    impact = f"Accuracy dropped by {(accuracy_before - accuracy_after) * 100:.2f}% due to poisoning."

    return {
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "impact": impact
    }