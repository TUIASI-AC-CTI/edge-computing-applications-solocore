import cv2
import mediapipe as mp
import gi
import numpy as np
import threading
import time
import math

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GdkPixbuf, GLib

DIST_TIP_DIP = 30
DIST_DIP_PIP = 30
DIST_TIP_PIP = 40
DRAW_BONES = True

FINGER_NAMES = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
FINGER_COLORS = {
    "THUMB": (255, 0, 0),
    "INDEX": (0, 0, 255),
    "MIDDLE": (0, 255, 0),
    "RING": (0, 255, 255),
    "PINKY": (255, 0, 255)
}
HANDS = ["LEFT", "RIGHT"]

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

active_keys = {}
key_last_seen = {}
for hand in HANDS:
    for finger in FINGER_NAMES:
        key_name = f"{hand}_{finger}"
        active_keys[key_name] = None
        key_last_seen[key_name] = 0

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

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def draw_keyboard(frame):
    key_positions = {}
    h, w, _ = frame.shape
    ref_w, ref_h = 1920, 1080
    scale = ((w / ref_w) + (h / ref_h)) / 2.0
    border_thickness = max(1, int(3 * scale))
    margin_x = int(30 * scale)
    margin_y = int(30 * scale)
    key_height = int(120 * scale)
    gap = int(3 * scale)
    row_y = margin_y

    for row in keys:
        num_keys = len(row)
        key_width = min(int(80 * scale),
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
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, 1)[0]
            text_x = row_x + (w_key - text_size[0]) // 2
            text_y = row_y + (key_height + text_size[1]) // 2
            cv2.putText(frame, key, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (255, 255, 255), 1)
            key_positions[key] = (row_x, row_y, row_x + w_key, row_y + key_height)
            row_x += w_key + gap
        row_y += key_height + gap
    return key_positions

class CameraWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Mediapipe Keyboard")
        self.set_default_size(800, 600)
        self.image = Gtk.Image()
        self.add(self.image)

        self.cap = cv2.VideoCapture(3)
        if not self.cap.isOpened():
            print("Wrong index")
            exit(1)

        self.latest_frame = None
        self.lock = threading.Lock()

        self.worker_thread = threading.Thread(target=self.process_frames)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def process_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            key_positions = draw_keyboard(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

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
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), FINGER_COLORS[finger], 2)
                                    print(f"{hand_label} {finger} → {key}")
                                    break
                            cv2.circle(frame, tip_pos, 10, FINGER_COLORS[finger], -1)

            with self.lock:
                self.latest_frame = frame.copy()

    def update_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return True
            frame = self.latest_frame.copy()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame_rgb.shape
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            frame_rgb.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            w,
            h,
            w * c,
        )
        self.image.set_from_pixbuf(pixbuf)
        return True

    def on_destroy(self, *args):
        self.cap.release()
        Gtk.main_quit()

if __name__ == "__main__":
    win = CameraWindow()
    win.connect("destroy", win.on_destroy)
    win.show_all()

    GLib.timeout_add(20, win.update_frame)
    Gtk.main()
