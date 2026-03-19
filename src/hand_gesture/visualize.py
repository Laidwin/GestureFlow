"""3D visualization of hand landmarks from JSON data."""

import argparse
import json

import matplotlib.pyplot as plt
import mediapipe as mp


def visualize_hand_3d(json_path: str, hand_index: int = 0):
    """Display a 3D scatter plot of hand landmarks from a JSON file."""
    mp_hands = mp.solutions.hands
    landmark_names = list(mp_hands.HandLandmark.__members__.keys())

    with open(json_path, "r") as f:
        data = json.load(f)

    hand_entry = data["result"][hand_index]
    landmarks_dict = list(hand_entry.values())[0]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"3D Hand Visualization (sample {hand_index})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.invert_yaxis()

    points_xyz = {}
    for name, coords in landmarks_dict.items():
        if not isinstance(coords, dict):
            continue
        points_xyz[name] = (coords["x"], coords["y"], coords["z"])
        ax.scatter(coords["x"], coords["y"], coords["z"], s=50, alpha=0.8)

    # Draw connections
    for connection in mp_hands.HAND_CONNECTIONS:
        try:
            start_name = connection[0].name
            end_name = connection[1].name
        except AttributeError:
            start_name = landmark_names[connection[0]]
            end_name = landmark_names[connection[1]]

        if start_name in points_xyz and end_name in points_xyz:
            s, e = points_xyz[start_name], points_xyz[end_name]
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], "r-")

    # Set equal aspect ratio
    all_coords = list(points_xyz.values())
    all_x = [p[0] for p in all_coords]
    all_y = [p[1] for p in all_coords]
    all_z = [p[2] for p in all_coords]

    max_range = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z),
    )
    mid_x = (max(all_x) + min(all_x)) / 2
    mid_y = (max(all_y) + min(all_y)) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize hand landmarks in 3D.")
    parser.add_argument("json_file", help="Path to JSON file with hand data")
    parser.add_argument(
        "--index", type=int, default=0, help="Hand sample index to plot"
    )
    args = parser.parse_args()
    visualize_hand_3d(args.json_file, args.index)


if __name__ == "__main__":
    main()
