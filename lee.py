import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from skimage.segmentation import slic
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading

# Simulated dataset loading (replace with actual dataset paths)
def load_data():
    data, labels, seg_masks = [], [], []
    img_size = (128, 128)

    landslide_path = "C:/Users/LEELA/OneDrive/Desktop/cutie/LANDSLIDES"
    no_landslide_path = "C:/Users/LEELA/OneDrive/Desktop/cutie/NO_LANDSLIDES"

    for img_file in os.listdir(landslide_path)[:10]:
        img = load_img(os.path.join(landslide_path, img_file), target_size=img_size)
        img = img_to_array(img) / 255.0
        data.append(img)
        labels.append(1)
        seg_masks.append(np.ones(img_size))

    for img_file in os.listdir(no_landslide_path)[:10]:
        img = load_img(os.path.join(no_landslide_path, img_file), target_size=img_size)
        img = img_to_array(img) / 255.0
        data.append(img)
        labels.append(0)
        seg_masks.append(np.zeros(img_size))

    return np.array(data), np.array(labels), np.array(seg_masks).reshape(-1, 128, 128, 1)

# CNN Model for Classification
def build_cnn_model(input_shape):
    inputs = Input(input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# U-Net Model for Segmentation
def build_unet_model(input_shape):
    inputs = Input(input_shape)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)

    u3 = UpSampling2D((2, 2))(c2)
    c3 = concatenate([u3, c1])
    c3 = Conv2D(16, (3, 3), activation='relu', padding='same')(c3)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c3)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# OBIA: Simple segmentation using SLIC
def obia_segmentation(image):
    segments = slic(image, n_segments=100, compactness=10, sigma=1)
    return segments

# Main function to process input image and predict
def landslide_detection(input_image_path, status_label):
    try:
        img_size = (128, 128)
        input_img = load_img(input_image_path, target_size=img_size)
        input_img_array = img_to_array(input_img) / 255.0
        input_img_array = np.expand_dims(input_img_array, axis=0)

        status_label.config(text="Loading dataset and training models...")
        X, y, seg_masks = load_data()
        X_train, X_test, y_train, y_test, seg_train, seg_test = train_test_split(X, y, seg_masks, test_size=0.2, random_state=42)

        status_label.config(text="Training CNN model...")
        cnn_model = build_cnn_model((128, 128, 3))
        cnn_model.fit(X_train, y_train, epochs=5, batch_size=4, validation_data=(X_test, y_test), verbose=0)

        status_label.config(text="Training U-Net model...")
        unet_model = build_unet_model((128, 128, 3))
        unet_model.fit(X_train, seg_train, epochs=5, batch_size=4, validation_data=(X_test, seg_test), verbose=0)

        status_label.config(text="Making predictions...")
        cnn_pred = cnn_model.predict(input_img_array)
        landslide_prob = cnn_pred[0][0]
        landslide_detected = landslide_prob > 0.5

        unet_pred = unet_model.predict(input_img_array)[0].squeeze()
        unet_pred_binary = (unet_pred > 0.5).astype(np.uint8) * 255

        obia_segments = obia_segmentation(input_img_array[0])

        # Display results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 4, 1)
        plt.title("Input Image")
        plt.imshow(input_img)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title(f"CNN Prediction: {'Landslide' if landslide_detected else 'No Landslide'} ({landslide_prob:.2f})")
        plt.imshow(input_img_array[0])
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("U-Net Segmentation")
        plt.imshow(unet_pred_binary, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title("Heatmap")
        sns.heatmap(unet_pred, cmap='coolwarm', cbar=True, alpha=0.8)
        plt.axis('off')
        plt.show()

        # Confidence Distribution
        cnn_test_preds = cnn_model.predict(X_test).flatten()
        plt.figure(figsize=(8, 5))
        sns.histplot(cnn_test_preds, bins=10, kde=True, color='blue')
        plt.axvline(0.5, color='red', linestyle='dashed', label="Decision Threshold")
        plt.xlabel("Landslide Probability")
        plt.ylabel("Frequency")
        plt.title("Model Confidence Distribution")
        plt.legend()
        plt.show()

        status_label.config(text=f"Landslide Probability: {landslide_prob:.2f}\nPrediction: {'Landslide Detected' if landslide_detected else 'No Landslide Detected'}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        status_label.config(text="Error occurred during processing.")

# GUI Class
class LandslideDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Landslide Detection System")
        self.root.geometry("400x300")

        self.label = tk.Label(root, text="Landslide Detection System", font=("Arial", 16))
        self.label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)

        self.run_btn = tk.Button(root, text="Run Detection", command=self.run_detection, state=tk.DISABLED)
        self.run_btn.pack(pady=10)

        self.status_label = tk.Label(root, text="Please upload an image to start.", wraplength=350)
        self.status_label.pack(pady=20)

        self.image_path = None

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.status_label.config(text=f"Image selected: {os.path.basename(self.image_path)}")
            self.run_btn.config(state=tk.NORMAL)

    def run_detection(self):
        if self.image_path:
            self.status_label.config(text="Processing image...")
            self.run_btn.config(state=tk.DISABLED)
            # Run detection in a separate thread to avoid freezing the GUI
            threading.Thread(target=landslide_detection, args=(self.image_path, self.status_label), daemon=True).start()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = LandslideDetectionApp(root)
    root.mainloop()