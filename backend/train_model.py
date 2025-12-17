import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

def train():
    df = pd.read_csv("raw_dataset_sample.csv")

    X = df[FEATURES]
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    calibrated_model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=5
    )

    calibrated_model.fit(X_train, y_train)

    joblib.dump(calibrated_model, os.path.join(MODEL_DIR, "failure_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("Models saved successfully")

if __name__ == "__main__":
    train()
