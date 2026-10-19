"""
Adult Census Income Analysis and Prediction
Module: model_training.py
Description: Builds, tunes, trains, and saves the Multilayer Perceptron (MLP) model
             using TensorFlow/Keras with early stopping and reproducible seeds.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def set_reproducible_seeds(seed: int = 42):
    """Sets random seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_mlp_model(
    input_dim: int,
    hidden_units: List[int] = [128, 64],
    dropout_rates: List[float] = [0.30, 0.20],
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Constructs the Multilayer Perceptron (MLP) architecture specified by the project:
    - Input Layer matching feature count
    - Dense(128, activation='relu')
    - Dropout(0.30)
    - Dense(64, activation='relu')
    - Dropout(0.20)
    - Dense(1, activation='sigmoid')

    Parameters
    ----------
    input_dim : int
        Dimension of preprocessed input feature vectors.
    hidden_units : List[int]
        Units in hidden dense layers. Default: [128, 64].
    dropout_rates : List[float]
        Dropout probabilities. Default: [0.30, 0.20].
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    keras.Model
        Compiled Keras sequential model.
    """
    model = keras.Sequential(name="Adult_Income_MLP")
    model.add(layers.Input(shape=(input_dim,)))

    for i, (units, drop) in enumerate(zip(hidden_units, dropout_rates)):
        model.add(layers.Dense(units, activation="relu", name=f"dense_layer_{i+1}"))
        if drop > 0:
            model.add(layers.Dropout(drop, name=f"dropout_{i+1}"))

    model.add(layers.Dense(1, activation="sigmoid", name="output_layer"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metrics_dir: str = "outputs/metrics"
) -> Dict[str, Any]:
    """
    Performs a practical hyperparameter comparison using ONLY validation data:
    Evaluates variations in hidden-layer sizes, dropout rates, and learning rates.
    Selects the best model configuration based on validation loss and AUC.
    """
    set_reproducible_seeds(42)
    input_dim = X_train.shape[1]

    candidate_configs = [
        {
            "name": "Config_1_Baseline",
            "hidden_units": [128, 64],
            "dropout_rates": [0.30, 0.20],
            "learning_rate": 0.001,
            "description": "Specified Assignment Baseline (128-64, drop 0.3/0.2, lr=1e-3)",
        },
        {
            "name": "Config_2_Compact",
            "hidden_units": [64, 32],
            "dropout_rates": [0.20, 0.10],
            "learning_rate": 0.001,
            "description": "Compact architecture (64-32, drop 0.2/0.1, lr=1e-3)",
        },
        {
            "name": "Config_3_Wider_LowLR",
            "hidden_units": [256, 128],
            "dropout_rates": [0.35, 0.25],
            "learning_rate": 0.0005,
            "description": "Wider architecture with lower learning rate (256-128, drop 0.35/0.25, lr=5e-4)",
        },
    ]

    results = []
    print("\n--- HYPERPARAMETER TUNING (VALIDATION SET ONLY) ---")

    for cfg in candidate_configs:
        print(f"\nEvaluating: {cfg['name']} ({cfg['description']})...")
        model = build_mlp_model(
            input_dim=input_dim,
            hidden_units=cfg["hidden_units"],
            dropout_rates=cfg["dropout_rates"],
            learning_rate=cfg["learning_rate"],
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=0,
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=128,
            callbacks=[early_stopping],
            verbose=0,
        )

        # In Keras 3, evaluate may return grouped compile_metrics, so compute metrics reliably:
        val_probs = model.predict(X_val, batch_size=256, verbose=0).ravel()
        val_preds = (val_probs >= 0.50).astype(int)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, log_loss

        val_loss = float(log_loss(y_val, np.clip(val_probs, 1e-7, 1 - 1e-7)))
        val_acc = float(accuracy_score(y_val, val_preds))
        val_prec = float(precision_score(y_val, val_preds, zero_division=0))
        val_rec = float(recall_score(y_val, val_preds, zero_division=0))
        val_auc = float(roc_auc_score(y_val, val_probs))

        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        res = {
            "Config Name": cfg["name"],
            "Hidden Units": str(cfg["hidden_units"]),
            "Dropout Rates": str(cfg["dropout_rates"]),
            "Learning Rate": cfg["learning_rate"],
            "Best Epoch": best_epoch,
            "Val Loss": round(val_loss, 4),
            "Val Accuracy": round(val_acc, 4),
            "Val Precision": round(val_prec, 4),
            "Val Recall": round(val_rec, 4),
            "Val AUC": round(val_auc, 4),
        }
        results.append(res)
        print(f"  Result -> Val Loss: {res['Val Loss']:.4f} | Val Accuracy: {res['Val Accuracy']:.4f} | Val AUC: {res['Val AUC']:.4f}")

    results_df = pd.DataFrame(results)
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "hyperparameter_tuning_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n[Hyperparameter Tuning] Results saved to: {results_file}")

    # Select best configuration by minimum validation loss (or highest AUC)
    best_row = results_df.sort_values(by="Val Loss", ascending=True).iloc[0]
    best_name = best_row["Config Name"]
    best_cfg = next(c for c in candidate_configs if c["name"] == best_name)
    print(f"[Hyperparameter Tuning] Selected Best Configuration: '{best_name}' with Val Loss = {best_row['Val Loss']} and Val AUC = {best_row['Val AUC']}")

    return {
        "comparison_table": results_df,
        "best_config": best_cfg,
    }


def train_production_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_config: Dict[str, Any],
    models_dir: str = "models",
    metrics_dir: str = "outputs/metrics",
    max_epochs: int = 20,
    batch_size: int = 128,
) -> Tuple[keras.Model, Dict[str, List[float]]]:
    """
    Trains the final model on training set with early stopping monitoring validation loss.
    Saves the best model weights and training history.
    """
    set_reproducible_seeds(42)
    models_path = Path(models_dir)
    metrics_path = Path(metrics_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    input_dim = X_train.shape[1]
    model = build_mlp_model(
        input_dim=input_dim,
        hidden_units=best_config.get("hidden_units", [128, 64]),
        dropout_rates=best_config.get("dropout_rates", [0.30, 0.20]),
        learning_rate=best_config.get("learning_rate", 0.001),
    )

    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    print(f"\n--- TRAINING MLP MODEL (Max Epochs: {max_epochs}, Batch Size: {batch_size}) ---")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Save model in standard .keras format
    model_save_path = models_path / "mlp_income_model.keras"
    model.save(model_save_path)
    print(f"[ModelTraining] Model successfully saved to: {model_save_path}")

    # Save history metrics
    serializable_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_file = metrics_path / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=4)
    print(f"[ModelTraining] Training history saved to: {history_file}")

    return model, serializable_history


if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessing import prepare_data_pipeline

    df, _ = load_dataset()
    data_dict = prepare_data_pipeline(df)

    tuning_out = tune_hyperparameters(
        data_dict["X_train"], data_dict["y_train"],
        data_dict["X_val"], data_dict["y_val"]
    )

    model, history = train_production_model(
        data_dict["X_train"], data_dict["y_train"],
        data_dict["X_val"], data_dict["y_val"],
        tuning_out["best_config"]
    )
    print("Model training completed successfully.")
