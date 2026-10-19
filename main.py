import sys
import os
import subprocess
from pathlib import Path

# Auto-detect and switch to .venv if current Python environment is missing TensorFlow
def _ensure_venv():
    try:
        import tensorflow
        return
    except ImportError:
        pass

    base = Path(__file__).resolve().parent
    candidates = [
        base.parent / ".venv" / "Scripts" / "python.exe",
        base / ".venv" / "Scripts" / "python.exe",
        Path(".venv/Scripts/python.exe").resolve(),
        Path("../.venv/Scripts/python.exe").resolve(),
    ]
    for py_exe in candidates:
        if py_exe.is_file():
            print(f"[Info] Current Python ({sys.executable}) lacks dependencies.")
            print(f"[Info] Automatically executing using project environment: {py_exe}\n")
            result = subprocess.run([str(py_exe)] + sys.argv, check=False)
            sys.exit(result.returncode)

    print("[Error] TensorFlow is not installed in the active Python environment.")
    print("Please activate the virtual environment:")
    print("  .\\.venv\\Scripts\\Activate.ps1")
    print("Or run using:")
    print("  .\\.venv\\Scripts\\python.exe main.py")
    sys.exit(1)

_ensure_venv()

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_dataset
from preprocessing import prepare_data_pipeline
from eda_analysis import run_comprehensive_eda
from visualizations import generate_all_eda_visualizations
from model_training import tune_hyperparameters, train_production_model
from evaluation import evaluate_test_set, plot_evaluation_charts, compute_permutation_importance


def main():
    print("=" * 80)
    print("ADULT CENSUS INCOME ANALYSIS & PREDICTION: END-TO-END PIPELINE")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "adult.csv"
    metrics_dir = base_dir / "outputs" / "metrics"
    figures_dir = base_dir / "outputs" / "figures"
    processed_dir = base_dir / "outputs" / "processed"
    models_dir = base_dir / "models"

    for d in [metrics_dir, figures_dir, processed_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. DATA LOADING & STRUCTURAL AUDIT
    print("\n>>> STEP 1: LOADING & PROFILING DATASET...")
    raw_df, dataset_summary = load_dataset(
        data_path=str(data_path),
        output_metrics_dir=str(metrics_dir)
    )

    # 2. EXPLORATORY DATA ANALYSIS (METRICS & TABLES)
    print("\n>>> STEP 2: RUNNING STATISTICAL EDA...")
    eda_results = run_comprehensive_eda(
        df=raw_df,
        metrics_dir=str(metrics_dir)
    )

    # 3. GENERATING EDA VISUALIZATIONS
    print("\n>>> STEP 3: RENDERING HIGH-RESOLUTION EDA VISUALIZATIONS...")
    generate_all_eda_visualizations(
        df=raw_df,
        figures_dir=str(figures_dir)
    )

    # 4. DATA CLEANING, IMPUTATION, ENCODING, & STRATIFIED SPLIT
    print("\n>>> STEP 4: PREPROCESSING & SPLITTING DATA (PREVENTING LEAKAGE)...")
    data_dict = prepare_data_pipeline(
        df=raw_df,
        outputs_dir=str(base_dir / "outputs")
    )

    # 5. HYPERPARAMETER TUNING ON VALIDATION SET
    print("\n>>> STEP 5: HYPERPARAMETER TUNING (VALIDATION SET ONLY)...")
    tuning_results = tune_hyperparameters(
        X_train=data_dict["X_train"],
        y_train=data_dict["y_train"],
        X_val=data_dict["X_val"],
        y_val=data_dict["y_val"],
        metrics_dir=str(metrics_dir)
    )

    # 6. TRAIN FINAL MULTILAYER PERCEPTRON (MLP)
    print("\n>>> STEP 6: TRAINING FINAL MLP MODEL WITH EARLY STOPPING...")
    model, history = train_production_model(
        X_train=data_dict["X_train"],
        y_train=data_dict["y_train"],
        X_val=data_dict["X_val"],
        y_val=data_dict["y_val"],
        best_config=tuning_results["best_config"],
        models_dir=str(models_dir),
        metrics_dir=str(metrics_dir),
        max_epochs=20,
        batch_size=128
    )

    # 7. MODEL EVALUATION ON HELD-OUT TEST SET
    print("\n>>> STEP 7: EVALUATING MODEL ON HELD-OUT TEST SET...")
    eval_results = evaluate_test_set(
        model=model,
        X_test=data_dict["X_test"],
        y_test=data_dict["y_test"],
        metrics_dir=str(metrics_dir)
    )

    # 8. GENERATING MODEL EVALUATION CHARTS
    print("\n>>> STEP 8: RENDERING MODEL EVALUATION CHARTS...")
    plot_evaluation_charts(
        history=history,
        y_test=data_dict["y_test"],
        y_prob=eval_results["y_prob"],
        y_pred=eval_results["y_pred"],
        figures_dir=str(figures_dir)
    )

    # 9. PERMUTATION FEATURE IMPORTANCE
    print("\n>>> STEP 9: COMPUTING PERMUTATION FEATURE IMPORTANCE...")
    compute_permutation_importance(
        model=model,
        X_val=data_dict["X_val"],
        y_val=data_dict["y_val"],
        feature_names=data_dict["feature_names"],
        figures_dir=str(figures_dir),
        metrics_dir=str(metrics_dir),
        n_repeats=3,
        top_k=15
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"All metrics saved to: {metrics_dir.resolve()}")
    print(f"All figures saved to: {figures_dir.resolve()}")
    print(f"Trained model saved to: {models_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
