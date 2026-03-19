"""Train a RandomForest classifier on fused gesture data."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_samples_from_file(filepath: str) -> np.ndarray:
    """Load hand samples from a fused JSON file and return a feature matrix."""
    with open(filepath, "r") as f:
        data = json.load(f)["result"]

    samples = []
    for entry in data:
        _, hand_points = next(iter(entry.items()))
        keys = list(hand_points.keys())
        coords = []
        # First 21 entries are landmark dicts with x/y/z
        for i in range(21):
            pt = hand_points[keys[i]]
            coords.extend([pt["x"], pt["y"], pt["z"]])
        # Remaining entries are scalar features
        for i in range(21, len(keys)):
            coords.append(hand_points[keys[i]])
        samples.append(coords)

    return np.array(samples)


def train(data_dir: str = "data/fused", model_dir: str = "models"):
    """Train a RandomForest model on all fused JSON files."""
    data_path = Path(data_dir)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    files = list(data_path.glob("*.json"))
    if not files:
        print(f"[ERROR] No JSON files found in {data_dir}")
        return

    X_parts, y_parts = [], []
    for file in files:
        samples = load_samples_from_file(str(file))
        label = file.stem  # e.g. "CloseFist_normalized"
        X_parts.append(samples)
        y_parts.extend([label] * len(samples))

    X = np.vstack(X_parts)
    y = np.array(y_parts)
    print(f"Loaded {len(X)} samples ({len(np.unique(y))} classes)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("RandomForest model trained successfully!")

    y_pred = rf.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (RandomForest)")
    plt.tight_layout()
    plt.show()

    output_file = model_path / "hand_gesture_rf.pkl"
    joblib.dump((rf, scaler), output_file)
    print(f"Model saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Train gesture recognition model.")
    parser.add_argument("--data-dir", default="data/fused")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    train(args.data_dir, args.model_dir)


if __name__ == "__main__":
    main()
