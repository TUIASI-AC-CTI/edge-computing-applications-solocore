import cv2
import mediapipe as mp
import numpy as np
import os

CAMERA_INDEX = 1
DATA_FILE = "hand_data.txt"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils


def save_data(label, landmarks):
    with open(DATA_FILE, "a") as f:
        data_row = [str(label)]
        for lm in landmarks.landmark:
            data_row.append(f"{lm.x:.6f}")
            data_row.append(f"{lm.y:.6f}")
            data_row.append(f"{lm.z:.6f}")

        f.write(",".join(data_row) + "\n")
    print(f"Saved sample for Label: {label}")


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}")
        return

    print(f"Recording to {DATA_FILE}")
    print("Instructions: Position your RIGHT hand and press 1, 2, or 3 to save data.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_handedness in enumerate(results.multi_handedness):
                if hand_handedness.classification[0].label == "Right":
                    current_right_hand = results.multi_hand_landmarks[idx]

                    mp_drawing.draw_landmarks(frame, current_right_hand, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, "RIGHT HAND READY", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3')]:
            label = chr(key)
            if current_right_hand:
                save_data(label, current_right_hand)
            else:
                print("Warning: No right hand detected! Point not saved.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()