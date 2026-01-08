import cv2
import numpy as np
import threading
import time
import math
import platform
import os
import sys

try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe is broken or not installed. Run: pip3 install mediapipe")
    sys.exit(1)

IS_WINDOWS = platform.system() == "Windows"
if not IS_WINDOWS:
    import gi

    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk, GdkPixbuf, GLib

CAM_WIDTH, CAM_HEIGHT = 640, 480
PROCESS_WIDTH, PROCESS_HEIGHT = 160, 120
TARGET_FPS = 30
DETECTION_SKIP_FRAMES = 2
TFLITE_MODEL_PATH = "hand_model.tflite"

DIST_TIP_DIP = 25
DIST_DIP_PIP = 25
DIST_TIP_PIP = 35
DRAW_BONES = True

FINGER_NAMES = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
FINGER_COLORS = {"THUMB": (255, 0, 0), "INDEX": (0, 0, 255), "MIDDLE": (0, 255, 0), "RING": (0, 255, 255),
                 "PINKY": (255, 0, 255)}

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


def load_interpreter():
    if IS_WINDOWS:
        return tflite.Interpreter(model_path=TFLITE_MODEL_PATH), "CPU (Laptop)"
    for path in ["/usr/lib/libvx_delegate.so", "/usr/lib/libnn_delegate.so", "/usr/lib/libvsi_npu_delegate.so"]:
        if os.path.exists(path):
            try:
                return tflite.Interpreter(model_path=TFLITE_MODEL_PATH,
                                          experimental_delegates=[tflite.load_delegate(path)]), "NPU (Accelerated)"
            except:
                pass
    return tflite.Interpreter(model_path=TFLITE_MODEL_PATH), "CPU (Fallback)"


interpreter, hw_status = load_interpreter()
interpreter.allocate_tensors()
input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()

mp_hands = mp.solutions.hands
FINGER_TIP_MAP = {"THUMB": 4, "INDEX": 8, "MIDDLE": 12, "RING": 16, "PINKY": 20}
FINGER_DIP_MAP = {"THUMB": 3, "INDEX": 7, "MIDDLE": 11, "RING": 15, "PINKY": 19}
FINGER_PIP_MAP = {"THUMB": 2, "INDEX": 6, "MIDDLE": 10, "RING": 14, "PINKY": 18}

keys = [
    ["ESC", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "BACKSPACE"],
    ["~", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "ENTER"],
    ["TAB", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]", "\\", "DEL"],
    ["CAPS", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "SHIFT"],
    ["SPACE"]
]
key_positions = {}


def create_keyboard_overlay(w, h):
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    scale = ((w / 1920) + (h / 1080)) / 2.0
    margin_x, margin_y = int(30 * scale), int(30 * scale)
    key_h, gap = int(200 * scale), int(3 * scale)
    row_y = margin_y
    for row in keys:
        n = len(row)
        kw = min(int(80 * scale), (w - 2 * margin_x - (n - 1) * gap) // n)
        row_w = n * kw + (n - 1) * gap
        row_x = (w - row_w) // 2
        for key in row:
            w_k = kw if key != "SPACE" else int(kw * 4)
            if key == "SPACE": row_x = (w - w_k) // 2
            x1, y1, x2, y2 = row_x, row_y, row_x + w_k, row_y + key_h
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), max(1, int(3 * scale)))
            cv2.putText(overlay, key, (x1 + 2, y1 + int(key_h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale,
                        (255, 255, 255), 1)
            key_positions[key] = (x1, y1, x2, y2)
            row_x += w_k + gap
        row_y += key_h + gap
    return overlay


keyboard_overlay = create_keyboard_overlay(CAM_WIDTH, CAM_HEIGHT)

mp_drawing = mp.solutions.drawing_utils
hands_proc = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0, min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)


class CameraWindow:
    def __init__(self):
        if not IS_WINDOWS:
            self.win = Gtk.Window(title="NXP Hybrid System")
            self.image_widget = Gtk.Image()
            self.win.add(self.image_widget)
            self.win.show_all()

        self.cap = cv2.VideoCapture(4)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        self.latest_frame = None
        self.left_tips = []
        self.last_results = None
        self.display_gesture = "None"
        self.right_coord = None
        self.typed_text = ""
        self.lock = threading.Lock()

        self.keys_pressed = set()
        self.last_right_gesture = 0

        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.process_loop, daemon=True).start()

    def capture_loop(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = cv2.flip(frame, 1)
            else:
                time.sleep(0.01)

    def process_loop(self):
        while True:
            time.sleep(0.001)
            with self.lock:
                if self.latest_frame is None: continue
                frame = self.latest_frame.copy()

            rgb_small = cv2.cvtColor(cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT)), cv2.COLOR_BGR2RGB)
            results = hands_proc.process(rgb_small)

            temp_left_tips = []
            temp_gesture_name = "None"
            temp_right_coord = None

            if results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label

                    if label == "Left":
                        for f in FINGER_NAMES:
                            t, d, p = hand_lms.landmark[FINGER_TIP_MAP[f]], hand_lms.landmark[FINGER_DIP_MAP[f]], \
                            hand_lms.landmark[FINGER_PIP_MAP[f]]
                            tp, dp, pp = (int(t.x * CAM_WIDTH), int(t.y * CAM_HEIGHT)), (int(d.x * CAM_WIDTH),
                                                                                         int(d.y * CAM_HEIGHT)), (
                                int(p.x * CAM_WIDTH), int(p.y * CAM_HEIGHT))
                            d1, d2, d3 = math.hypot(tp[0] - dp[0], tp[1] - dp[1]), math.hypot(dp[0] - pp[0], dp[1] - pp[
                                1]), math.hypot(tp[0] - pp[0], tp[1] - pp[1])

                            if d1 < DIST_TIP_DIP and d2 < DIST_DIP_PIP and d3 < DIST_TIP_PIP:
                                temp_left_tips.append((tp, FINGER_COLORS[f]))

                    elif label == "Right":
                        inp = []
                        for lm in hand_lms.landmark: inp.extend([lm.x, lm.y, lm.z])
                        interpreter.set_tensor(input_details[0]['index'], np.array([inp], dtype=np.float32))
                        interpreter.invoke()
                        gid = np.argmax(interpreter.get_tensor(output_details[0]['index'])[0]) + 1

                        gesture_map = {1: "COORD MODE", 2: "CLICK", 3: "RIGHT CLICK"}
                        temp_gesture_name = gesture_map.get(gid, "Unknown")
                        it = hand_lms.landmark[8]
                        temp_right_coord = (int(it.x * CAM_WIDTH), int(it.y * CAM_HEIGHT))

                        if gid != self.last_right_gesture:
                            if gid == 2:
                                print("CLICK")
                            elif gid == 3:
                                print("RIGHT CLICK")
                            self.last_right_gesture = gid

            if not any(h.classification[0].label == "Right" for h in
                       results.multi_handedness) if results.multi_handedness else True:
                self.last_right_gesture = 0

            with self.lock:
                self.left_tips = temp_left_tips
                self.display_gesture = temp_gesture_name
                self.right_coord = temp_right_coord
                self.last_results = results

    def render(self):
        with self.lock:
            if self.latest_frame is None: return True
            display, tips, res, gesture_text, r_coord = self.latest_frame.copy(), self.left_tips, self.last_results, self.display_gesture, self.right_coord

        display = cv2.addWeighted(display, 1.0, keyboard_overlay, 0.3, 0)

        cv2.putText(display, f"HW: {hw_status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(display, f"RIGHT GESTURE: {gesture_text}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if gesture_text == "COORD MODE" and r_coord:
            cv2.circle(display, r_coord, 7, (0, 255, 0), -1)
            cv2.putText(display, f"X:{r_coord[0]} Y:{r_coord[1]}", (r_coord[0] + 15, r_coord[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if res and res.multi_hand_landmarks:
            for idx, hl in enumerate(res.multi_hand_landmarks):
                mp_drawing.draw_landmarks(display, hl, mp_hands.HAND_CONNECTIONS)
                label = res.multi_handedness[idx].classification[0].label
                wrist = hl.landmark[0]
                cv2.putText(display, label, (int(wrist.x * CAM_WIDTH), int(wrist.y * CAM_HEIGHT) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cp = set()
        for p, c in tips:
            cv2.circle(display, p, 8, c, -1)
            for k, (x1, y1, x2, y2) in key_positions.items():
                if x1 < p[0] < x2 and y1 < p[1] < y2:
                    cv2.rectangle(display, (x1, y1), (x2, y2), c, 3)
                    cp.add(k)

        new_k = cp - self.keys_pressed
        for k in new_k:
            print(k, end='', flush=True)
            if k == "BACKSPACE":
                self.typed_text = self.typed_text[:-1]
            elif k == "SPACE":
                self.typed_text += " "
            elif len(k) == 1:
                self.typed_text += k
        self.keys_pressed = cp

        cv2.rectangle(display, (0, CAM_HEIGHT - 40), (CAM_WIDTH, CAM_HEIGHT), (0, 0, 0), -1)
        cv2.putText(display, f"TYPED: {self.typed_text[-30:]}", (10, CAM_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        if IS_WINDOWS:
            cv2.imshow("NXP Hybrid Console", display)
            return cv2.waitKey(1) & 0xFF != ord('q')
        else:
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            pix = GdkPixbuf.Pixbuf.new_from_data(rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * c)
            self.image_widget.set_from_pixbuf(pix)
            return True


if __name__ == "__main__":
    app = CameraWindow()
    if IS_WINDOWS:
        while app.render(): pass
    else:
        GLib.timeout_add(int(1000 / TARGET_FPS), app.render)
        Gtk.main()