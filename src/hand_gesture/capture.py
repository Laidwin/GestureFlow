"""Capture hand landmarks from webcam and save them as JSON."""

import argparse
import json
import time

import cv2
import mediapipe as mp
import numpy as np

CAMERA_INDEX = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Capture hand gesture landmarks.")
    parser.add_argument("--gesture", required=True, help="Gesture label (e.g. CloseFist)")
    parser.add_argument("--name", required=True, help="Name of the person capturing")
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory for JSON files (default: data/raw)",
    )
    return parser.parse_args()


def run(gesture: str, name: str, output_dir: str) -> None:
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{gesture}_{name}.json"

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    # Load existing data if the file already exists
    counter = 0
    if output_file.exists():
        with open(output_file, "r") as f:
            content = f.read()
            if content:
                data = json.loads(content)
                if data.get("result"):
                    last_key = list(data["result"][-1].keys())[0]
                    counter = int(last_key.split("_")[-1])

    json_data = {"result": []}
    if output_file.exists():
        with open(output_file, "r") as f:
            content = f.read()
            if content:
                json_data = json.loads(content)

    saved_count = 0
    landmark_names = list(mp_hands.HandLandmark.__members__.keys())

    print(f"[INFO] Capture started (ESC to quit, SPACE to save).")
    print(f"[INFO] Gesture: {gesture} | Person: {name}")

    try:
        while True:
            ret, image = cap.read()
            if not ret:
                print("[ERROR] Video stream interrupted.")
                break

            frame = cv2.flip(image, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_points = None
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    hand_points = np.array([[lm.x, lm.y] for lm in hand_lms.landmark])

            cv2.putText(
                frame,
                f"Label: {gesture}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Capture gesture", frame)

            if hand_points is not None:
                canvas = np.ones((400, 400, 3), dtype=np.uint8) * 20
                pts = (hand_points * 400).astype(int)
                for x, y in pts:
                    cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Landmark Map", canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("[INFO] Capture stopped by user.")
                break

            elif key == 32:  # SPACE
                if results.multi_hand_world_landmarks:
                    hand = results.multi_hand_world_landmarks[0]
                    counter += 1
                    hand_data = {
                        f"{gesture}_{name}_{counter}": {
                            lm_name: {"x": lm.x, "y": lm.y, "z": lm.z}
                            for lm_name, lm in zip(landmark_names, hand.landmark)
                        }
                    }
                    json_data["result"].append(hand_data)
                    saved_count += 1
                    print(f"[OK] Saved sample for '{gesture}', total: {saved_count}")
                else:
                    print("[WARNING] No hand detected.")
                time.sleep(0.25)
    finally:
        # Save on exit
        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=2)
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Data saved to {output_file}")


def main():
    args = parse_args()
    run(gesture=args.gesture, name=args.name, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
