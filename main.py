"""Main entrypoint for Stock Price Trend Prediction."""

from src.dataset import get_stock_dataset
from src.model import build_model, train_model, evaluate_model


def run_pipeline():
    print("=" * 50)
    print("  Stock Price Trend Prediction Pipeline")
    print("=" * 50)

    print("\n[1/3] Loading and engineering stock features...")
    X_train, X_test, y_train, y_test = get_stock_dataset()
    print(f"      Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"      Features: {list(X_train.columns)}")

    print("\n[2/3] Training trend classification model...")
    model = build_model()
    train_model(model, X_train, y_train)

    print("\n[3/3] Evaluating model on unseen future test window...")
    results = evaluate_model(model, X_test, y_test)
    print(f"      Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    for label, metrics in results["classification_report"].items():
        if isinstance(metrics, dict):
            print(f"  Class {label} (0=Down, 1=Up): Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")


if __name__ == "__main__":
    run_pipeline()
