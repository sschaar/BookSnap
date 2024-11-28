import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Definiere die Zuordnung von Klassenindex zu Klassenname
class_names = {0: "Berserk", 1: "Monster", 2: "Neon Genesis Evangelion"}

# Basispfad zum Modellverzeichnis und Angabe der Modellversion
model_base_path = '../models/trained_model'
model_version = '2'  # Hier kannst du die gewünschte Version anpassen
model_path = os.path.join(model_base_path, f'book_cover_classification_model_v{model_version}.h5')

# Prüfen, ob das Modell existiert
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Modell laden
model = tf.keras.models.load_model(model_path)
print(f"Loaded model version: {model_version} from {model_path}")

# Bildparameter definieren
img_height = 180
img_width = 180

# Funktion zur Vorhersage
def predict_image_class(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalisieren
    img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])  # Klasse mit höchster Wahrscheinlichkeit
    predicted_class_name = class_names[predicted_index]  # Klassenname basierend auf Index
    return predicted_class_name

# Pfad zum Ordner mit den Inferenzbildern
inference_data_dir = '../data/inference_data'

# Durch alle Ordner und Bilder iterieren und Vorhersagen ausgeben
for folder_name in os.listdir(inference_data_dir):
    folder_path = os.path.join(inference_data_dir, folder_name)
    if os.path.isdir(folder_path):  # Nur Ordner berücksichtigen
        print(f"\nPredictions for folder '{folder_name}':")
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            predicted_class_name = predict_image_class(image_path)
            print(f"Image: {image_name} - Predicted Class: {predicted_class_name}")
