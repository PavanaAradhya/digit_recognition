import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint    # ✅ ADD THIS LINE

# -------- Load Dataset --------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# -------- Build Model --------
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------- Checkpoint Path --------
checkpoint_path = "digit_model.keras"
checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1)

# -------- Train Model --------
model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[checkpoint])

# -------- Final Save --------
model.save("final_digit_model.keras")
