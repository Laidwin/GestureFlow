"""Real-time hand gesture prediction using webcam and trained model."""

import argparse

import cv2
import joblib
import mediapipe as mp
import numpy as np

from hand_gesture.normalize import normalize_hand_from_tuples

CONFIDENCE_THRESHOLD = 0.5


def run(model_path: str = "models/hand_gesture_rf.pkl"):
    """Run real-time gesture recognition from webcam."""
    rf, scaler = joblib.load(model_path)
    labels = rf.classes_

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        label = "No gesture"
        confidence = 0.0

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                normalized = normalize_hand_from_tuples(landmarks)

                # Flatten: 21 xyz points + 15 scalar features
                vec = []
                for i in range(21):
                    vec.extend([normalized[i][0], normalized[i][1], normalized[i][2]])
                for i in range(21, len(normalized)):
                    vec.append(normalized[i])

                vec = np.array(vec).reshape(1, -1)
                vec_scaled = scaler.transform(vec)

                probs = rf.predict_proba(vec_scaled)[0]
                max_idx = np.argmax(probs)
                confidence = probs[max_idx]

                label = labels[max_idx] if confidence >= CONFIDENCE_THRESHOLD else "No gesture"
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        text = f"{label} ({confidence * 100:.1f}%)" if label != "No gesture" else label
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time gesture prediction.")
    parser.add_argument(
        "--model", default="models/hand_gesture_rf.pkl", help="Path to trained model"
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
