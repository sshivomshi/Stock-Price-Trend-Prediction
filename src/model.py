from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def build_model(n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
    """Build and return a baseline Random Forest model."""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


def train_model(model: RandomForestClassifier, X_train, y_train) -> RandomForestClassifier:
    """Train the model on the provided training data."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test, y_test) -> dict:
    """Evaluate model performance and return metrics."""
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    return {
        "accuracy": acc,
        "classification_report": report,
    }
