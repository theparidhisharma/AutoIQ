import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type"
]

TARGET = "Machine failure"

def train():
    df = pd.read_csv("AI4I_dataset.csv")
    print(df.columns)

    df = df.drop(["UDI", "Product ID"], axis=1, errors="ignore")
    df["Type"] = df["Type"].map({"L":0, "M":1, "H":2})

    X = df[FEATURES]
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KEEP test data — do not discard it
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight={0: 1, 1: 6},
        random_state=42
    )

    calibrated_model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=5
    )

    # TRAIN FIRST
    calibrated_model.fit(X_train, y_train)

    # EVALUATE AFTER TRAINING
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]

    threshold = 0.30
    y_pred = (y_prob >= threshold).astype(int)

    print("=== AutoIQ Model Evaluation ===")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC Score:")
    print(roc_auc_score(y_test, y_prob))

    # SAVE MODELS
    joblib.dump(calibrated_model, os.path.join(MODEL_DIR, "failure_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("Models saved successfully")

if __name__ == "__main__":
    train()
