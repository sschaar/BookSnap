import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Modell laden
model = load_model('book_cover_recognition_model.h5')

# Pfad zum Ordner mit den neuen Bildern
image_folder = 'test'

# Bildgröße (muss dieselbe Größe sein, die für das Training verwendet wurde)
img_width, img_height = 180, 180 # Beispielgröße, passe dies an dein Modell an


# Funktion, um ein Bild zu laden und vorzuverarbeiten
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen
    img_array /= 255.0  # Normalisierung (falls im Training gemacht)
    return img_array


# Alle Bilder im Ordner durchgehen und Vorhersagen treffen
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)

    # Überprüfen, ob es eine Bilddatei ist
    if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
        img_array = load_and_preprocess_image(img_path)

        # Vorhersage mit dem geladenen Modell
        predictions = model.predict(img_array)

        # Ausgabe der Vorhersage
        predicted_class = np.argmax(predictions, axis=1)
        print(f'Bild: {img_name}, Vorhergesagte Klasse: {predicted_class}')
