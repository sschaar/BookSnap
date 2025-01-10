import os
import sys
import base64
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Umgebungsvariablen für TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Modell-Konfiguration
class_names = {0: "Berserk", 1: "Monster", 2: "Neon Genesis Evangelion"}
model_base_path = '/root/rately-files/models/'
model_version = '2'
model_path = f"{model_base_path}/book_cover_classification_model_v{model_version}.h5"

model = load_model(model_path)

img_height = 180
img_width = 180

# Ecken erkennen und markieren
def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_contour_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and area > largest_contour_area:
            largest_contour = approx
            largest_contour_area = area

    if largest_contour is not None:
        pts = largest_contour.reshape(4, 2)
        for i, point in enumerate(pts):
            cv2.circle(image, tuple(point.astype(int)), 4, (0, 255, 0), -1)
            next_point = pts[(i + 1) % 4]
            cv2.line(image, tuple(point.astype(int)), tuple(next_point.astype(int)), (0, 255, 0), 1)

    return image

# Bild vorverarbeiten und Klasse vorhersagen
def predict_image_class(base64_string):
    try:
        # Base64-String dekodieren
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = np.array(image)

        # Ecken erkennen und markieren
        processed_image = detect_corners(image)

        # Bild für das Modell vorbereiten
        resized_image = cv2.resize(processed_image, (img_width, img_height))
        img_array = resized_image / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Vorhersage
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        return class_names[predicted_index]
    except Exception as e:
        return f"Error processing image: {e}"

if __name__ == "__main__":
    base64_string = sys.stdin.read().strip()
    prediction = predict_image_class(base64_string)
    print(prediction)
