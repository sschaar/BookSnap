import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Schritt 1: Pfad zum Ordner, in dem die Bilder gespeichert sind
image_folder = 'images'  # Pfad zu den strukturierten Bildern
test_folder = 'test'      # Pfad zu den Testbildern

# Schritt 2: Bildpfade und Titel aus dem Ordner extrahieren
def load_images_from_folder(image_folder):
    image_paths = []
    titles = []

    # Alle Ordner im Hauptordner durchgehen
    for class_name in os.listdir(image_folder):
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):  # Sicherstellen, dass es ein Ordner ist
            # Alle Bilder in diesem Ordner durchgehen
            for filename in os.listdir(class_folder):
                if filename.endswith('.jpg'):  # Nur JPG-Dateien
                    image_path = os.path.join(class_folder, filename)
                    image_paths.append(image_path)
                    titles.append(class_name)  # Titel entspricht dem Ordnernamen

    return image_paths, titles

# Schritt 3: Bilder laden und Pfade sowie Titel extrahieren
image_paths, titles = load_images_from_folder(image_folder)

# Schritt 4: Labels in numerische Werte umwandeln
class_names = sorted(set(titles))  # Einzigartige Klassen
class_indices = {name: index for index, name in enumerate(class_names)}  # Mapping von Titeln zu Indizes
labels = [class_indices[title] for title in titles]  # Labels umwandeln

# Schritt 5: Labels in One-Hot-Encoded Format umwandeln
num_classes = len(class_names)  # Anzahl der einzigartigen Titel
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

# Schritt 6: Datensatz erstellen
img_height = 180  # Höhe der Bilder
img_width = 180   # Breite der Bilder
batch_size = 32   # Batch-Größe

# Schritt 7: ImageDataGenerator für Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Schritt 8: Erstellen des Generators
train_generator = datagen.flow_from_directory(
    image_folder,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Schritt 9: Modell definieren
model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Softmax-Aktivierung für Mehrklassenklassifikation
])

# Schritt 10: Modell kompilieren
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Verlusttyp ändern
              metrics=['accuracy'])

# Schritt 11: Modell-Zusammenfassung anzeigen
model.summary()

# Schritt 12: Training des Modells
epochs = 400
history = model.fit(train_generator, epochs=epochs)

# Schritt 13: Ergebnisse anzeigen
acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')

plt.show()

# Schritt 14: Modell speichern
model.save('book_cover_recognition_model.h5')

# Schritt 15: Vorhersagen für Testbilder
def predict_titles(test_folder):
    for class_name in os.listdir(test_folder):
        class_folder = os.path.join(test_folder, class_name)

        # Überprüfen, ob es ein Ordner ist
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)

                # Überprüfen, ob es eine Bilddatei ist
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalisierung
                    img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen

                    predictions = model.predict(img_array)
                    predicted_class_index = np.argmax(predictions, axis=1)[0]
                    predicted_title = class_names[predicted_class_index]  # Titel zurückgeben

                    print(f'Bild: {img_name}, Vorhergesagter Titel: {predicted_title} (Ordner: {class_name})')

# Vorhersagen für die Testbilder durchführen
predict_titles(test_folder)
