import cv2
import mediapipe as mp
import math
import time
from pynput.keyboard import Controller, Key

PRESS_THRESHOLD = 0.12
RELEASE_THRESHOLD = 0.15
MISS_THRESHOLD = 0.1

DIST_TIP_DIP = 30
DIST_DIP_PIP = 30
DIST_TIP_PIP = 40
DRAW_BONES = True

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

FINGER_NAMES = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
FINGER_COLORS = {
    "THUMB": (255, 0, 0),
    "INDEX": (0, 0, 255),
    "MIDDLE": (0, 255, 0),
    "RING": (0, 255, 255),
    "PINKY": (255, 0, 255)
}

active_keys = {}
key_start_time = {}
key_last_seen = {}

HANDS = ["LEFT", "RIGHT"]
for hand in HANDS:
    for finger in FINGER_NAMES:
        key_name = f"{hand}_{finger}"
        active_keys[key_name] = None
        key_start_time[key_name] = 0
        key_last_seen[key_name] = 0

keyboard = Controller()

SPECIAL_KEYS = {
    "SPACE": Key.space,
    "ENTER": Key.enter,
    "BACKSPACE": Key.backspace,
    "TAB": Key.tab,
    "SHIFT": Key.shift,
    "CAPS": Key.caps_lock,
    "ESC": Key.esc,
    "DEL": Key.delete,
    "F1": Key.f1, "F2": Key.f2, "F3": Key.f3, "F4": Key.f4,
    "F5": Key.f5, "F6": Key.f6, "F7": Key.f7, "F8": Key.f8,
    "F9": Key.f9, "F10": Key.f10, "F11": Key.f11, "F12": Key.f12
}

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def draw_keyboard(frame):
    key_positions = {}
    h, w, _ = frame.shape
    ref_w, ref_h = 3840, 2160
    scale_x = w / ref_w
    scale_y = h / ref_h
    scale = (scale_x + scale_y) / 2.0
    border_thickness = max(2, int(6 * scale))
    margin_x = int(50 * scale)
    margin_y = int(50 * scale)
    key_height = int(360 * scale)
    gap = int(4 * scale)
    row_y = margin_y
    width_multiplier = 2.0
    for row in keys:
        num_keys = len(row)
        key_width = min(int(110 * scale * width_multiplier),
                        (w - 2 * margin_x - (num_keys - 1) * gap) // num_keys)
        row_width = num_keys * key_width + (num_keys - 1) * gap
        row_x = (w - row_width) // 2
        for key in row:
            w_key = key_width
            if key == "SPACE":
                w_key = int(key_width * 4)
                row_x = (w - w_key) // 2
            cv2.rectangle(frame, (row_x, row_y), (row_x + w_key, row_y + key_height),
                          (255, 255, 255), border_thickness)
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, int(2 * scale))[0]
            text_x = row_x + (w_key - text_size[0]) // 2
            text_y = row_y + (key_height + text_size[1]) // 2
            cv2.putText(frame, key, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), int(2 * scale))
            key_positions[key] = (row_x, row_y, row_x + w_key, row_y + key_height)
            row_x += w_key + gap
        row_y += key_height + gap
    return key_positions, scale

FINGER_TIP = {
    "THUMB": mp_hands.HandLandmark.THUMB_TIP,
    "INDEX": mp_hands.HandLandmark.INDEX_FINGER_TIP,
    "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    "RING": mp_hands.HandLandmark.RING_FINGER_TIP,
    "PINKY": mp_hands.HandLandmark.PINKY_TIP
}

FINGER_DIP = {
    "THUMB": mp_hands.HandLandmark.THUMB_IP,
    "INDEX": mp_hands.HandLandmark.INDEX_FINGER_DIP,
    "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
    "RING": mp_hands.HandLandmark.RING_FINGER_DIP,
    "PINKY": mp_hands.HandLandmark.PINKY_PIP
}

FINGER_PIP = {
    "THUMB": mp_hands.HandLandmark.THUMB_MCP,
    "INDEX": mp_hands.HandLandmark.INDEX_FINGER_PIP,
    "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    "RING": mp_hands.HandLandmark.RING_FINGER_PIP,
    "PINKY": mp_hands.HandLandmark.PINKY_PIP
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    key_positions, scale = draw_keyboard(frame)
    current_time = time.time()
    detected_keys = {hand: {finger: None for finger in FINGER_NAMES} for hand in HANDS}
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[hand_idx].classification[0].label.upper()
            if DRAW_BONES:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h_frame, w_frame, _ = frame.shape
            for finger in FINGER_NAMES:
                tip = hand_landmarks.landmark[FINGER_TIP[finger]]
                dip = hand_landmarks.landmark[FINGER_DIP[finger]]
                pip = hand_landmarks.landmark[FINGER_PIP[finger]]
                tip_pos = (int(tip.x * w_frame), int(tip.y * h_frame))
                dip_pos = (int(dip.x * w_frame), int(dip.y * h_frame))
                pip_pos = (int(pip.x * w_frame), int(pip.y * h_frame))
                dist1 = distance(tip_pos, dip_pos)
                dist2 = distance(dip_pos, pip_pos)
                dist3 = distance(tip_pos, pip_pos)
                if dist1 < DIST_TIP_DIP and dist2 < DIST_DIP_PIP and dist3 < DIST_TIP_PIP:
                    for key, (x1, y1, x2, y2) in key_positions.items():
                        if x1 < tip_pos[0] < x2 and y1 < tip_pos[1] < y2:
                            detected_keys[hand_label][finger] = key
                            key_last_seen[f"{hand_label}_{finger}"] = current_time
                            cv2.rectangle(frame, (x1, y1), (x2, y2), FINGER_COLORS[finger], 3)
                            break
                    cv2.circle(frame, tip_pos, int(10 * scale), FINGER_COLORS[finger], -1)
    for hand in HANDS:
        for finger in FINGER_NAMES:
            key_name = f"{hand}_{finger}"
            detected_key = detected_keys[hand][finger]
            if active_keys[key_name]:
                if detected_key != active_keys[key_name] and (current_time - key_last_seen[key_name]) > RELEASE_THRESHOLD:
                    key_to_release = active_keys[key_name]
                    if key_to_release in SPECIAL_KEYS:
                        keyboard.release(SPECIAL_KEYS[key_to_release])
                    else:
                        keyboard.release(key_to_release)
                    print(f"{key_name} Released {key_to_release}")
                    active_keys[key_name] = None
                    key_start_time[key_name] = 0
            else:
                if detected_key:
                    if key_start_time[key_name] == 0:
                        key_start_time[key_name] = current_time
                    elif (current_time - key_start_time[key_name]) >= PRESS_THRESHOLD:
                        active_keys[key_name] = detected_key
                        key_to_press = active_keys[key_name]
                        if key_to_press in SPECIAL_KEYS:
                            keyboard.press(SPECIAL_KEYS[key_to_press])
                        else:
                            keyboard.press(key_to_press)
                        print(f"{key_name} Pressed {active_keys[key_name]}")
                else:
                    key_start_time[key_name] = 0
    cv2.imshow("Multi-Finger Two Hands Keyboard With Input", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
