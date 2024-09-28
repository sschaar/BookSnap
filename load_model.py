import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Modell laden
model = load_model('book_cover_recognition_model.h5')

# Pfad zum Ordner mit den neuen Bildern
image_folder = 'test'

# Bildgröße (muss dieselbe Größe sein, die für das Training verwendet wurde)
img_width, img_height = 180, 180  # Beispielgröße, passe dies an dein Modell an

# Liste der Buch-Titel, geordnet nach Index der Klassenlabels
book_titles = [
    "ALICE IN BORDERLAND Band 1",
    "ALICE IN BORDERLAND Band 2",
    "ALICE IN BORDERLAND Band 3",
    "ATTACK ON TITAN Band 1",
    "ATTACK ON TITAN Band 2",
    "ATTACK ON TITAN Band 3",
    "BERSERK MAX Band 1",
    "BERSERK MAX Band 2",
    "BERSERK MAX Band 3",
    "CHAINSAW MAN Band 1",
    "CHAINSAW MAN Band 2",
    "CHAINSAW MAN Band 3",
    "Die Zwerge",
    "EINE KURZE GESCHICHTE DER ZEIT",
    "HOMUNCULUS Band 1",
    "HOMUNCULUS Band 2",
    "HOMUNCULUS Band 3",
    "KIJIN GENTOSHO Band 1",
    "KIJIN GENTOSHO Band 2",
    "MEHR ZEIT",
    "MONSTER Band 1",
    "MONSTER Band 2",
    "MONSTER Band 3",
    "NEON GENESIS EVANGELION [PERFECT EDITION] Band 01",
    "NEON GENESIS EVANGELION [PERFECT EDITION] Band 02",
    "NEON GENESIS EVANGELION [PERFECT EDITION] Band 03",
    "SAKAMOTO DAYS Band 1",
    "SAKAMOTO DAYS Band 2",
    "SAKAMOTO DAYS Band 3",
    "THE ONE THING",
    "TOKYO GHOUL Band 1",
    "TOKYO GHOUL Band 2",
    "TOKYO GHOUL Band 3",
    "VINLAND SAGA Band 1",
    "VINLAND SAGA Band 2",
    "VINLAND SAGA Band 3"
]

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

        # Index der vorhergesagten Klasse
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Vorhergesagter Buchtitel
        predicted_book_title = book_titles[predicted_class]

        # Ausgabe der Vorhersage
        print(f'Bild: {img_name}, Vorhergesagter Buchtitel: {predicted_book_title}')
