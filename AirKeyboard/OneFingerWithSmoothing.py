import cv2
import mediapipe as mp
import math
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

keys = [
    ["ESC","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","BACKSPACE"],
    ["~","1","2","3","4","5","6","7","8","9","0","-","=","ENTER"],
    ["TAB","Q","W","E","R","T","Y","U","I","O","P","[","]","\\","DEL"],
    ["CAPS","A","S","D","F","G","H","J","K","L",";","'"],
    ["Z","X","C","V","B","N","M",",",".","/","SHIFT"],
    ["SPACE"]
]

active_key = None
key_start_time = 0
key_last_seen = 0

PRESS_THRESHOLD = 0.12
RELEASE_THRESHOLD = 0.15
MISS_THRESHOLD = 0.1

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def draw_keyboard(frame):
    key_positions = {}
    h, w, _ = frame.shape

    margin_x = 50
    margin_y = 50
    key_height = 120
    gap = 4

    row_y = margin_y
    for row in keys:
        num_keys = len(row)
        key_width = min(110, (w - 2 * margin_x - (num_keys - 1) * gap) // num_keys)
        row_width = num_keys * key_width + (num_keys - 1) * gap
        row_x = (w - row_width) // 2

        for key in row:
            w_key = key_width
            if key == "SPACE":
                w_key = key_width * 4
                row_x = (w - w_key) // 2
            cv2.rectangle(frame, (row_x, row_y), (row_x + w_key, row_y + key_height), (255,255,255), 2)
            cv2.putText(frame, key, (row_x + 10, row_y + key_height//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            key_positions[key] = (row_x, row_y, row_x + w_key, row_y + key_height)
            row_x += w_key + gap
        row_y += key_height + gap

    return key_positions

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    key_positions = draw_keyboard(frame)
    current_time = time.time()
    finger_key = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h_frame, w_frame, _ = frame.shape
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

            tip_pos = (int(tip.x * w_frame), int(tip.y * h_frame))
            dip_pos = (int(dip.x * w_frame), int(dip.y * h_frame))
            pip_pos = (int(pip.x * w_frame), int(pip.y * h_frame))

            dist1 = distance(tip_pos, dip_pos)
            dist2 = distance(dip_pos, pip_pos)
            dist3 = distance(tip_pos, pip_pos)

            if dist1 < 60 and dist2 < 60 and dist3 < 80:
                for key, (x1, y1, x2, y2) in key_positions.items():
                    if x1 < tip_pos[0] < x2 and y1 < tip_pos[1] < y2:
                        finger_key = key
                        key_last_seen = current_time
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
                        break
                cv2.circle(frame, tip_pos, 10, (0,0,255), -1)

    if active_key:
        if finger_key != active_key and (current_time - key_last_seen) > RELEASE_THRESHOLD:
            print(f"{active_key} Released")
            active_key = None
            key_start_time = 0
    else:
        if finger_key:
            if key_start_time == 0:
                key_start_time = current_time
            elif (current_time - key_start_time) >= PRESS_THRESHOLD:
                active_key = finger_key
                print(f"{active_key} Pressed")
        else:
            key_start_time = 0

    cv2.imshow("Smoothed Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
