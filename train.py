"""Train the bank churn ANN and save the model, preprocessor and evaluation report.

Usage:
    python train.py                 # default settings
    python train.py --epochs 150    # override training length
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Churn_Modelling.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

CATEGORICAL = ["Geography", "Gender"]
NUMERIC = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "Exited"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )


def build_model(n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = build_model(X_train_t.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=15, restore_best_weights=True
    )
    history = model.fit(
        X_train_t,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    proba = model.predict(X_test_t, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "baseline_accuracy": round(float(1 - y_test.mean()), 4),
        "epochs_trained": len(history.history["loss"]),
        "test_size": int(len(y_test)),
    }
    print(classification_report(y_test, pred, target_names=["Stayed", "Exited"]))
    print(json.dumps(metrics, indent=2))

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "churn_model.keras")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred), display_labels=["Stayed", "Exited"]
    ).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Confusion matrix (test set)")
    axes[1].plot(history.history["auc"], label="train")
    axes[1].plot(history.history["val_auc"], label="validation")
    axes[1].set_title("ROC-AUC per epoch")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "evaluation.png", dpi=120)
    print(f"Saved model to {MODEL_DIR} and report to {REPORT_DIR}")


if __name__ == "__main__":
    main()
