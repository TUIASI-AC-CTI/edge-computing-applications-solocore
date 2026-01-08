import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
\
INPUT_FILE = "my_hand_data.txt"
TFLITE_MODEL = "hand_model.tflite"


def train():
    data, labels = [], []
    expected_length = 64

    print(f"Reading {INPUT_FILE}...")

    with open(INPUT_FILE, "r") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split(",")

            try:
                row = [float(x) for x in parts if x.strip()]

                if len(row) == expected_length:
                    labels.append(int(row[0]) - 1)
                    data.append(row[1:])
                else:
                    print(f"Skipping line {line_num}: Found {len(row)} values, expected {expected_length}")
            except ValueError:
                print(f"Skipping line {line_num}: Contains non-numeric data.")

    if not data:
        print("Error: No valid data found to train on.")
        return

    X = np.array(data, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print(f"Successfully loaded {len(X)} valid samples.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("\nStarting Training...")
    model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test))

    print("\nConverting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(TFLITE_MODEL, 'wb') as f:
        f.write(tflite_model)

    print(f"Success! Model saved as: {TFLITE_MODEL}")


if __name__ == "__main__":
    train()