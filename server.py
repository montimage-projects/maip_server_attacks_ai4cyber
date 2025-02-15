from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
import os
import pandas as pd
import joblib
from model_inference import ModelInference, generate_poisoned_dataset, retrain_model, perform_random_label_swap_attack, perform_target_label_flip_attack
import json
import uuid
from flask import jsonify
from fastapi.responses import JSONResponse, FileResponse, Response

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

@app.post("/upload")
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
    model_path = os.path.join(model_folder, "model.h5")
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


@app.get("/models", response_model=dict)
async def list_models():
    """List all models in the uploads directory."""
    uploads_dir = os.path.join(UPLOAD_FOLDER)

    # Check if the uploads directory exists
    if not os.path.exists(uploads_dir):
        raise HTTPException(status_code=404, detail="Uploads directory not found.")

    # List all subdirectories in the uploads directory
    model_folders = [name for name in os.listdir(uploads_dir) if os.path.isdir(os.path.join(uploads_dir, name))]

    return {"models": model_folders}


@app.get("/models/{model_id}/train")
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


@app.get("/models/{model_id}/test")
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


@app.get("/evaluate")
async def evaluate_model(model_id: str):
    """Evaluate model accuracy & confusion matrix for the specified model."""
    # Load the specified model based on model_id
    model_folder = os.path.join(UPLOAD_FOLDER, model_id)

    # Check if the model folder exists
    if not os.path.exists(model_folder):
        return {"error": "Model not found."}

    # Load the model, scaler, and encoder from the specified folder
    model_path = os.path.join(model_folder, "model.h5")
    scaler_path = os.path.join(model_folder, "scaler.joblib")
    encoder_path = os.path.join(model_folder, "encoder.joblib")

    # Ensure the test data path is correct
    test_data_path = os.path.join(model_folder, "test.csv")  # Use the model folder for test.csv

    global model_inference
    model_inference = ModelInference(model_path, scaler_path, encoder_path)

    # Proceed with evaluation
    test_data = pd.read_csv(test_data_path)
    X_test_scaled, y_test = model_inference.preprocess_data(test_data)

    results = model_inference.predict(test_data)
    predictions = results["predictions"]

    accuracy = (predictions == y_test).mean()
    cm = pd.crosstab(y_test, predictions, rownames=["Actual"], colnames=["Predicted"])

    return {
        "model_id": model_id,
        "accuracy": accuracy,
        "confusion_matrix": cm.values.tolist()
    }


@app.post("/attacks/poisoning/ctgan")
async def apply_ctgan_poisoning(
    model_id: str,
    poisoning_rate: str,
    target_class: str = None,
    ctgan_file: str = None
):
    """Apply CTGAN poisoning attack to the training dataset of the specified model."""
    poisoning_rate_float = float(poisoning_rate)
    return await apply_poisoning(model_id, "ctgan", poisoning_rate_float, target_class, ctgan_file)


@app.post("/attacks/poisoning/random-swapping-labels")
async def apply_random_swapping_labels_poisoning(
    model_id: str,
    poisoning_rate: str
):
    """Apply random swapping labels poisoning attack to the training dataset of the specified model."""
    poisoning_rate_float = float(poisoning_rate)
    return await apply_poisoning(model_id, "rsl", poisoning_rate_float)


@app.post("/attacks/poisoning/target-label-flipping")
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
    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]

    # Convert poisoning rate from percentage to decimal for calculations
    poisoning_rate_decimal = poisoning_rate / 100.0

    poisoned_data_filename = "poisoned_data.csv"

    # Perform poisoning based on the attack type
    try:
        if attack_type == "ctgan":
            poisoned_X, poisoned_y = generate_poisoned_dataset(
                X_train.values.tolist(), y_train.tolist(),
                attack_type, poisoning_rate_decimal, target_class
            )
            # Create a descriptive filename for the poisoned data for CTGAN
            poisoning_rate_int = int(poisoning_rate)
            poisoned_data_filename = f"poisoned_train_ctgan_rate_{poisoning_rate_int}.csv"
        elif attack_type == "rsl":
            poisoned_X, poisoned_y = perform_random_label_swap_attack(X_train.values.tolist(), y_train.tolist(), poisoning_rate_decimal)
            # Create a descriptive filename for the poisoned data for Random Swapping Labels
            poisoning_rate_int = int(poisoning_rate)
            poisoned_data_filename = f"poisoned_train_rsl_rate_{poisoning_rate_int}.csv"
        elif attack_type == "tlf":
            poisoned_X, poisoned_y = perform_target_label_flip_attack(X_train.values.tolist(), y_train.tolist(), poisoning_rate_decimal, target_class)

            # Create a descriptive filename for the poisoned data including target class
            poisoning_rate_int = int(poisoning_rate)
            poisoned_data_filename = f"poisoned_train_tlf_rate_{poisoning_rate_int}_class_{target_class}.csv"
        else:
            raise ValueError("Invalid attack type.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create a descriptive filename for the poisoned data
    poisoned_data_path = os.path.join(model_folder, poisoned_data_filename)
    poisoned_df = pd.DataFrame(poisoned_X, columns=X_train.columns)
    poisoned_df["Label"] = poisoned_y
    poisoned_df.to_csv(poisoned_data_path, index=False)

    return {
        "message": f"{attack_type.replace('-', ' ').title()} poisoning attack applied",
        "poisoning_rate": poisoning_rate_int,
        "poisoned_data_path": poisoned_data_path
    }


@app.get("/models/{model_id}/poisoned-datasets")
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


@app.get("/models/{model_id}/poisoned-datasets/{dataset_name}")
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


@app.post("/retrain")
async def retrain(model_id: str, poisoned_data_filename: str):
    """Retrain the model on poisoned data & evaluate impact."""
    model_folder = os.path.join(UPLOAD_FOLDER, model_id)

    # Check if the model folder exists
    if not os.path.exists(model_folder):
        return {"error": "Model not found."}

    # Load the model, scaler, and encoder from the specified folder
    model_path = os.path.join(model_folder, "model.h5")
    scaler_path = os.path.join(model_folder, "scaler.joblib")
    encoder_path = os.path.join(model_folder, "encoder.joblib")

    global model_inference
    model_inference = ModelInference(model_path, scaler_path, encoder_path)

    # Load the original test data for evaluation
    test_data_path = os.path.join(model_folder, "test.csv")
    if not os.path.exists(test_data_path):
        return {"error": f"Test data file not found: {test_data_path}"}

    test_data = pd.read_csv(test_data_path)
    X_test_scaled, y_test = model_inference.preprocess_data(test_data)

    # Evaluate before retraining
    results = model_inference.predict(test_data)
    predictions = results["predictions"]
    accuracy_before = (predictions == y_test).mean()
    cm_before = pd.crosstab(y_test, predictions).values.tolist()

    # Load poisoned dataset
    poisoned_data_path = os.path.join(model_folder, poisoned_data_filename)
    if not os.path.exists(poisoned_data_path):
        return {"error": f"Poisoned dataset not found: {poisoned_data_path}"}

    # Call the retrain_model function
    accuracy_after, cm_after = retrain_model(model_inference, poisoned_data_path, test_data_path)

    impact = f"Accuracy dropped by {(accuracy_before - accuracy_after) * 100:.2f}% due to poisoning."

    return {
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "confusion_matrix_before": cm_before,
        "confusion_matrix_after": cm_after,
        "impact": impact,
        "poisoned_data_path": poisoned_data_path
    }

