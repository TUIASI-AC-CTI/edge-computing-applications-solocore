import cv2
import mediapipe as mp
import numpy as np
import sys
import platform

IS_WINDOWS = platform.system() == "Windows"
if not IS_WINDOWS:
    import gi

    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk, GdkPixbuf, GLib

TFLITE_MODEL = "hand_model.tflite"
CAMERA_INDEX = 2

import tensorflow.lite as tflite

try:
    if not IS_WINDOWS:
        delegate = tflite.load_delegate("/usr/lib/libvsi_npu_delegate.so")
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL, experimental_delegates=[delegate])
    else:
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
except Exception as e:
    print(f"NPU Delegate not found, falling back to CPU: {e}")
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


class HandApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.latest_prediction = "None"

    def predict(self, landmarks):
        input_data = []
        for lm in landmarks.landmark:
            input_data.extend([lm.x, lm.y, lm.z])

        input_tensor = np.array([input_data], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return np.argmax(output_data[0]) + 1

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, handedness in enumerate(results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    lm = results.multi_hand_landmarks[idx]
                    label = self.predict(lm)
                    cv2.putText(frame, f"CLASS: {label}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        return frame


app = HandApp()

if IS_WINDOWS:
    while True:
        img = app.process_frame()
        if img is not None:
            cv2.imshow("Laptop Predictor", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
else:
    window = Gtk.Window(title="NXP NPU Predictor")
    window.connect("destroy", Gtk.main_quit)
    image_widget = Gtk.Image()
    window.add(image_widget)
    window.show_all()


    def update_gtk():
        frame = app.process_frame()
        if frame is not None:
            h, w, c = frame.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(frame.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * c)
            image_widget.set_from_pixbuf(pixbuf)
        return True


    GLib.timeout_add(30, update_gtk)
    Gtk.main()