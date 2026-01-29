from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

MODEL_OUT = Path("models")
MODEL_FILE = MODEL_OUT / "model.joblib"


def main():
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    data = load_iris()
    X, y = data.data, data.target
    # Simple random forest for demo
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, MODEL_FILE)
    print(f"Model trained and saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()