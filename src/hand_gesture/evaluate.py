"""Evaluate the classifier on people it never saw (leave-one-person-out)."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from hand_gesture.train import load_samples_from_file


def evaluate(data_dir: str = "data/normalized"):
    """Train on every person but one, test on that person, for each person."""
    files = sorted(Path(data_dir).glob("*.json"))
    if not files:
        print(f"[ERROR] No JSON files found in {data_dir}")
        return

    X_parts, labels, people = [], [], []
    for file in files:
        # Files are named <Gesture>_<Person>_normalized.json
        gesture, person = file.stem.split("_")[:2]
        samples = load_samples_from_file(str(file))
        X_parts.append(samples)
        labels.extend([gesture] * len(samples))
        people.extend([person] * len(samples))

    X = np.vstack(X_parts)
    y = np.array(labels)
    groups = np.array(people)
    y_pred = np.empty_like(y)

    for person in np.unique(groups):
        test = groups == person
        scaler = StandardScaler().fit(X[~test])
        rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
        rf.fit(scaler.transform(X[~test]), y[~test])
        y_pred[test] = rf.predict(scaler.transform(X[test]))
        accuracy = (y_pred[test] == y[test]).mean()
        print(f"{person:<10} {test.sum():>4} samples  accuracy {accuracy * 100:.1f}%")

    errors = int((y_pred != y).sum())
    print(f"\nOverall: {(y_pred == y).mean() * 100:.1f}% ({errors} errors out of {len(y)} samples)")

    classes = np.unique(y)
    print("\nConfusion matrix (rows: actual, columns: predicted):")
    print(" " * 10 + "".join(f"{c:>10}" for c in classes))
    for name, row in zip(classes, confusion_matrix(y, y_pred, labels=classes)):
        print(f"{name:<10}" + "".join(f"{v:>10}" for v in row))


def main():
    parser = argparse.ArgumentParser(description="Leave-one-person-out evaluation.")
    parser.add_argument("--data-dir", default="data/normalized")
    args = parser.parse_args()
    evaluate(args.data_dir)


if __name__ == "__main__":
    main()
