"""Fuse normalized JSON files by gesture label into single files per gesture."""

import argparse
import json
from pathlib import Path

GESTURES = ["CloseFist", "OkSign", "OpenHand", "ThumbsUp", "IndexUp"]


def fuse_gesture(
    gesture: str, input_dir: str = "data/normalized", output_dir: str = "data/fused"
):
    """Merge all normalized JSON files matching a gesture keyword."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fused_data = {"result": []}

    for file_path in input_path.glob("*.json"):
        if gesture in file_path.name:
            with open(file_path, "r") as f:
                data = json.load(f)
            fused_data["result"].extend(data.get("result", []))

    output_file = output_path / f"{gesture}_normalized.json"
    with open(output_file, "w") as f:
        json.dump(fused_data, f, indent=2)

    print(f"Fused {len(fused_data['result'])} samples -> {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Fuse normalized gesture data.")
    parser.add_argument("--input-dir", default="data/normalized")
    parser.add_argument("--output-dir", default="data/fused")
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=GESTURES,
        help=f"Gesture labels to fuse (default: {GESTURES})",
    )
    args = parser.parse_args()

    for gesture in args.gestures:
        fuse_gesture(gesture, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
