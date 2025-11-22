import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("leaf_model.h5")

# Preprocess function
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict():
    global img_path

    if img_path == "":
        result_label.config(text="Please select an image first!")
        return

    img_array = preprocess(img_path)
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result_label.config(text="Prediction: HEALTHY LEAF 🌿", fg="green")
    else:
        result_label.config(text="Prediction: DISEASED LEAF ⚠️", fg="red")

# Choose image function
def choose_image():
    global img_path
    img_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )

    if img_path:
        img = Image.open(img_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)

        image_label.config(image=img)
        image_label.image = img  # keep reference

        result_label.config(text="")  # clear previous result

# GUI Window
root = tk.Tk()
root.title("Leaf Disease Detection")
root.geometry("400x450")
root.resizable(False, False)

img_path = ""

title = tk.Label(root, text="Leaf Disease Classifier", font=("Arial", 18, "bold"))
title.pack(pady=10)

btn = tk.Button(root, text="Select Leaf Image", command=choose_image, font=("Arial", 13))
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

predict_btn = tk.Button(root, text="Predict", command=predict, font=("Arial", 13))
predict_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=10)

root.mainloop()
