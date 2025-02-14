from fastapi import FastAPI, UploadFile, File, Form
import os
import pandas as pd
import joblib
from model_inference import ModelInference, generate_poisoned_dataset, retrain_model
import json

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

    # Save files
    with open(MODEL_PATH, "wb") as f:
        f.write(model.file.read())
    with open(TRAIN_DATA_PATH, "wb") as f:
        f.write(train_data.file.read())
    with open(TEST_DATA_PATH, "wb") as f:
        f.write(test_data.file.read())

    if scaler:
        with open(SCALER_PATH, "wb") as f:
            f.write(scaler.file.read())
    if encoder:
        with open(ENCODER_PATH, "wb") as f:
            f.write(encoder.file.read())

    global model_inference
    model_inference = ModelInference(MODEL_PATH, SCALER_PATH, ENCODER_PATH)

    return {"message": "Files uploaded successfully", "model_path": MODEL_PATH}


@app.get("/evaluate")
async def evaluate_model():
    """Evaluate model accuracy & confusion matrix."""
    if not model_inference:
        return {"error": "Model not loaded. Upload files first."}

    test_data = pd.read_csv(TEST_DATA_PATH)
    X_test_scaled, y_test = model_inference.preprocess_data(test_data)

    results = model_inference.predict(test_data)
    predictions = results["predictions"]

    accuracy = (predictions == y_test).mean()
    cm = pd.crosstab(y_test, predictions, rownames=["Actual"], colnames=["Predicted"])

    return {"accuracy": accuracy, "confusion_matrix": cm.values.tolist()}


@app.post("/attack")
async def apply_poisoning(
    attack_type: str = Form(...),
    poisoning_rate: float = Form(...),
    target_class: str = Form(None),
    ctgan_file: UploadFile = File(None)
):
    """Apply a poisoning attack to the training dataset."""
    if not model_inference:
        return {"error": "Model not loaded. Upload files first."}

    train_data = pd.read_csv(TRAIN_DATA_PATH)
    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]

    # Perform poisoning
    poisoned_X, poisoned_y = generate_poisoned_dataset(
        X_train.values.tolist(), y_train.tolist(),
        attack_type, poisoning_rate, target_class
    )

    # Save poisoned data
    poisoned_data_path = os.path.join(UPLOAD_FOLDER, "poisoned_train.csv")
    poisoned_df = pd.DataFrame(poisoned_X, columns=X_train.columns)
    poisoned_df["Label"] = poisoned_y
    poisoned_df.to_csv(poisoned_data_path, index=False)

    return {
        "message": "Poisoning attack applied",
        "attack_type": attack_type,
        "poisoning_rate": poisoning_rate,
        "poisoned_data_path": poisoned_data_path
    }


@app.post("/retrain")
async def retrain():
    """Retrain the model on poisoned data & evaluate impact."""
    if not model_inference:
        return {"error": "Model not loaded. Upload files first."}

    # Load original test data for evaluation
    test_data = pd.read_csv(TEST_DATA_PATH)
    X_test_scaled, y_test = model_inference.preprocess_data(test_data)

    # Evaluate before retraining
    results = model_inference.predict(test_data)
    predictions = results["predictions"]
    accuracy_before = (predictions == y_test).mean()
    cm_before = pd.crosstab(y_test, predictions).values.tolist()

    # Retrain on poisoned dataset
    poisoned_data_path = os.path.join(UPLOAD_FOLDER, "poisoned_train.csv")
    retrain_model(model_inference, poisoned_data_path, TEST_DATA_PATH)

    # Evaluate after retraining
    results = model_inference.predict(test_data)
    predictions = results["predictions"]
    accuracy_after = (predictions == y_test).mean()
    cm_after = pd.crosstab(y_test, predictions).values.tolist()

    impact = f"Accuracy dropped by {(accuracy_before - accuracy_after) * 100:.2f}% due to poisoning."

    return {
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "confusion_matrix_before": cm_before,
        "confusion_matrix_after": cm_after,
        "impact": impact
    }
