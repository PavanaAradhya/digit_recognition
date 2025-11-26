import tkinter as tk
from tkinter import *
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# Load trained model
model = tf.keras.models.load_model("model/final_digit_model.keras")

# Window
root = tk.Tk()
root.title("Digit Recognition")
root.geometry("400x500")
root.configure(bg="white")

# Canvas for drawing
canvas_width = 280
canvas_height = 280
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg="black")
canvas.pack(pady=20)

image1 = Image.new("L", (canvas_width, canvas_height), "black")
draw = ImageDraw.Draw(image1)

def paint(event):
    x1, y1 = (event.x - 6), (event.y - 6)
    x2, y2 = (event.x + 6), (event.y + 6)
    canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
    draw.ellipse([x1, y1, x2, y2], fill="white")

canvas.bind("<B1-Motion>", paint)

result_label = Label(root, text="Draw a digit", font=("Arial", 16), bg="white")
result_label.pack()

def predict_digit():
    img = image1.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    result_label.config(text=f"Predicted Digit: {digit}")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, canvas_width, canvas_height), fill="black")
    result_label.config(text="Draw a digit")

Button(root, text="Predict", command=predict_digit, font=("Arial", 14)).pack(pady=10)
Button(root, text="Clear", command=clear_canvas, font=("Arial", 14)).pack()

root.mainloop()
