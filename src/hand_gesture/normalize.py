"""Normalize hand landmark data relative to wrist and palm orientation."""

import argparse
import json
from pathlib import Path

import numpy as np

# Finger angle/length feature definitions (landmark indices)
FINGER_ANGLES = {
    "THUMB_ANGLE": (2, 3, 4),
    "INDEX_ANGLE": (5, 6, 8),
    "MIDDLE_FINGER_ANGLE": (9, 10, 12),
    "RING_FINGER_ANGLE": (13, 14, 16),
    "PINKY_ANGLE": (17, 18, 20),
}

FINGER_LENGTHS = {
    "THUMB_LENGTH": (4, 2),
    "INDEX_LENGTH": (8, 5),
    "MIDDLE_FINGER_LENGTH": (12, 9),
    "RING_FINGER_LENGTH": (16, 13),
    "PINKY_LENGTH": (20, 17),
}

INTER_FINGER_FEATURES = {
    "THUMB_INDEX_ANGLE": (4, 2, 8),
    "THUMB_INDEX_DISTANCE": (6, 4),
    "THUMB_INDEX_X": (6, 4, 0),
    "THUMB_INDEX_Y": (7, 5, 1),
    "THUMB_INDEX_Z": (8, 6, 2),
}


def angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the angle (in degrees) at point b formed by segments ba and bc."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _build_rotation_matrix(wrist, ring_mcp, index_mcp):
    """Build a rotation matrix to align the hand to a canonical orientation."""
    z_axis = ring_mcp - wrist
    z_axis /= np.linalg.norm(z_axis)

    x_temp = ring_mcp - index_mcp
    x_axis = x_temp - np.dot(x_temp, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _compute_extra_features(points: np.ndarray) -> dict:
    """Compute additional geometric features (angles, lengths, distances)."""
    features = {}

    for name, (a, b, c) in FINGER_ANGLES.items():
        features[name] = angle_3d(points[a], points[b], points[c])

    for name, (tip, base) in FINGER_LENGTHS.items():
        features[name] = float(np.linalg.norm(points[tip] - points[base]))

    features["THUMB_INDEX_ANGLE"] = angle_3d(
        points[4], points[2], points[8]
    )
    features["THUMB_INDEX_DISTANCE"] = float(
        np.linalg.norm(points[6] - points[4])
    )
    features["THUMB_INDEX_X"] = float(points[6][0] - points[4][0])
    features["THUMB_INDEX_Y"] = float(points[7][1] - points[5][1])
    features["THUMB_INDEX_Z"] = float(points[8][2] - points[6][2])

    return features


def normalize_hand(hand_dict: dict) -> dict:
    """Normalize a hand landmark dictionary (from JSON) with rotation and scaling.

    Returns a dict with normalized xyz coords per landmark + extra features.
    """
    points = np.array([[v["x"], v["y"], v["z"]] for v in hand_dict.values()])
    keys = list(hand_dict.keys())

    wrist = np.array(list(hand_dict["WRIST"].values()))
    ring_mcp = np.array(list(hand_dict["RING_FINGER_MCP"].values()))
    index_mcp = np.array(list(hand_dict["INDEX_FINGER_MCP"].values()))

    distance = np.linalg.norm(ring_mcp - wrist)
    if distance == 0:
        distance = 1.0

    points_norm = points / distance
    wrist_norm = wrist / distance
    ring_norm = ring_mcp / distance
    index_norm = index_mcp / distance

    R = _build_rotation_matrix(wrist_norm, ring_norm, index_norm)

    oriented = [(R.T @ (lm - wrist_norm)) for lm in points_norm]

    result = {
        key: {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
        for key, p in zip(keys, oriented)
    }
    result.update(_compute_extra_features(points_norm))
    return result


def normalize_hand_from_tuples(hand_tuples: list) -> list:
    """Normalize hand data given as a list of (x, y, z) tuples (from MediaPipe live).

    Returns a flat list: 21 rotated [x, y, z] arrays + 15 scalar features.
    """
    points = np.array([[v[0], v[1], v[2]] for v in hand_tuples])

    wrist = np.array(hand_tuples[0])
    ring_mcp = np.array(hand_tuples[13])
    index_mcp = np.array(hand_tuples[5])

    distance = np.linalg.norm(ring_mcp - wrist)
    if distance == 0:
        distance = 1.0

    points_norm = points / distance
    wrist_norm = wrist / distance
    ring_norm = ring_mcp / distance
    index_norm = index_mcp / distance

    R = _build_rotation_matrix(wrist_norm, ring_norm, index_norm)

    oriented = [(R.T @ (lm - wrist_norm)) for lm in points_norm]

    extra = _compute_extra_features(points_norm)
    return oriented + list(extra.values())


def process_files(input_dir: str = "data/raw", output_dir: str = "data/normalized"):
    """Normalize all JSON files in input_dir and write results to output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for json_file in input_path.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        normalized_data = {"result": []}
        for entry in data["result"]:
            new_entry = {}
            for name, hand in entry.items():
                new_entry[name] = normalize_hand(hand)
            normalized_data["result"].append(new_entry)

        out_file = output_path / f"{json_file.stem}_normalized.json"
        with open(out_file, "w") as f:
            json.dump(normalized_data, f, indent=2)

        print(f"Normalized: {json_file.name} -> {out_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Normalize hand landmark JSON files.")
    parser.add_argument("--input-dir", default="data/raw", help="Raw JSON directory")
    parser.add_argument(
        "--output-dir", default="data/normalized", help="Output directory"
    )
    args = parser.parse_args()
    process_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
