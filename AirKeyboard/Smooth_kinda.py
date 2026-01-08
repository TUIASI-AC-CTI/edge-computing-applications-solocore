import cv2
import mediapipe as mp
import gi
import numpy as np
import threading
import time
import math

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GdkPixbuf, GLib

CAM_WIDTH, CAM_HEIGHT = 640, 480
PROCESS_WIDTH, PROCESS_HEIGHT = 160, 120
TARGET_FPS = 30
DETECTION_SKIP_FRAMES = 3
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

keys = [
    ["ESC","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","BACKSPACE"],
    ["~","1","2","3","4","5","6","7","8","9","0","-","=","ENTER"],
    ["TAB","Q","W","E","R","T","Y","U","I","O","P","[","]","\\","DEL"],
    ["CAPS","A","S","D","F","G","H","J","K","L",";","'"],
    ["Z","X","C","V","B","N","M",",",".","/","SHIFT"],
    ["SPACE"]
]
key_positions = {}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

FINGER_TIP = { "THUMB": mp_hands.HandLandmark.THUMB_TIP,
               "INDEX": mp_hands.HandLandmark.INDEX_FINGER_TIP,
               "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               "RING": mp_hands.HandLandmark.RING_FINGER_TIP,
               "PINKY": mp_hands.HandLandmark.PINKY_TIP }

FINGER_DIP = { "THUMB": mp_hands.HandLandmark.THUMB_IP,
               "INDEX": mp_hands.HandLandmark.INDEX_FINGER_DIP,
               "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
               "RING": mp_hands.HandLandmark.RING_FINGER_DIP,
               "PINKY": mp_hands.HandLandmark.PINKY_PIP }

FINGER_PIP = { "THUMB": mp_hands.HandLandmark.THUMB_MCP,
               "INDEX": mp_hands.HandLandmark.INDEX_FINGER_PIP,
               "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
               "RING": mp_hands.HandLandmark.RING_FINGER_PIP,
               "PINKY": mp_hands.HandLandmark.PINKY_PIP }

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def create_keyboard_overlay(frame_w, frame_h):
    overlay = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    ref_w, ref_h = 1920, 1080
    scale = ((frame_w/ref_w)+(frame_h/ref_h))/2.0
    border_thickness = max(1, int(3*scale))
    margin_x = int(30*scale)
    margin_y = int(30*scale)
    key_height = int(200*scale)
    gap = int(3*scale)
    row_y = margin_y

    for row in keys:
        num_keys = len(row)
        key_width = min(int(80*scale), (frame_w-2*margin_x-(num_keys-1)*gap)//num_keys)
        row_width = num_keys*key_width + (num_keys-1)*gap
        row_x = (frame_w-row_width)//2
        for key in row:
            w_key = key_width
            if key=="SPACE":
                w_key = int(key_width*4)
                row_x = (frame_w-w_key)//2
            x1, y1 = row_x, row_y
            x2, y2 = row_x+w_key, row_y+key_height
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (255,255,255), border_thickness)
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.7*scale, 1)[0]
            text_x = x1 + (w_key - text_size[0])//2
            text_y = y1 + (key_height + text_size[1])//2
            cv2.putText(overlay, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7*scale, (255,255,255), 1)
            key_positions[key] = (x1, y1, x2, y2)
            row_x += w_key + gap
        row_y += key_height + gap
    return overlay

keyboard_overlay = create_keyboard_overlay(CAM_WIDTH, CAM_HEIGHT)

class CameraWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Mediapipe Keyboard Optimized")
        self.set_default_size(CAM_WIDTH, CAM_HEIGHT)
        self.image = Gtk.Image()
        self.add(self.image)

        self.cap = cv2.VideoCapture(3, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit(1)

        self.latest_frame = None
        self.fingertips = []
        self.last_hand_landmarks = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.keys_pressed = set()

        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.process_loop, daemon=True).start()

    def capture_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame.copy()

    def process_loop(self):
        while True:
            time.sleep(0.001)
            self.frame_count += 1
            if self.frame_count % DETECTION_SKIP_FRAMES != 0:
                continue

            with self.lock:
                if self.latest_frame is None:
                    continue
                frame = self.latest_frame.copy()

            small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_small)

            fingertips = []
            landmarks = None
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = results.multi_handedness[idx].classification[0].label.upper()
                    for finger in FINGER_NAMES:
                        tip = hand_landmarks.landmark[FINGER_TIP[finger]]
                        dip = hand_landmarks.landmark[FINGER_DIP[finger]]
                        pip = hand_landmarks.landmark[FINGER_PIP[finger]]

                        tip_pos = (int(tip.x*CAM_WIDTH), int(tip.y*CAM_HEIGHT))
                        dip_pos = (int(dip.x*CAM_WIDTH), int(dip.y*CAM_HEIGHT))
                        pip_pos = (int(pip.x*CAM_WIDTH), int(pip.y*CAM_HEIGHT))

                        dist1 = distance(tip_pos, dip_pos)
                        dist2 = distance(dip_pos, pip_pos)
                        dist3 = distance(tip_pos, pip_pos)
                        if dist1 < DIST_TIP_DIP and dist2 < DIST_DIP_PIP and dist3 < DIST_TIP_PIP:
                            fingertips.append((tip_pos, FINGER_COLORS[finger]))

            with self.lock:
                self.fingertips = fingertips
                self.last_hand_landmarks = landmarks

    def update_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return True
            frame = self.latest_frame.copy()

            alpha = 0.3
            display_frame = cv2.addWeighted(frame, 1.0, keyboard_overlay, alpha, 0)

            if self.last_hand_landmarks and DRAW_BONES:
                for hand_landmarks in self.last_hand_landmarks:
                    mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            current_pressed = set()
            for tip_pos, color in self.fingertips:
                cv2.circle(display_frame, tip_pos, 8, color, -1)
                for key, (x1,y1,x2,y2) in key_positions.items():
                    if x1<tip_pos[0]<x2 and y1<tip_pos[1]<y2:
                        cv2.rectangle(display_frame, (x1,y1),(x2,y2), color, 3)
                        current_pressed.add(key)
                        break

            new_keys = current_pressed - self.keys_pressed
            for k in new_keys:
                print(k, end='', flush=True)
            self.keys_pressed = current_pressed

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame_rgb.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                frame_rgb.tobytes(),
                GdkPixbuf.Colorspace.RGB,
                False,
                8,
                w,
                h,
                w*c
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
    GLib.timeout_add(int(1000/TARGET_FPS), win.update_frame)
    Gtk.main()
