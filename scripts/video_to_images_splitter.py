import cv2
import os
import random

# Pfade festlegen
video_folder = ('./data/raw/video')  # Ordner mit den Videos
train_folder = './data/processed/images/train_data'  # Hauptordner für Trainingsbilder
test_folder = './data/processed/images/test_data'  # Hauptordner für Testbilder

# Einstellungen
frame_skip = 10  # Anzahl der zu überspringenden Frames
max_frames_per_video = 500  # Maximale Anzahl der Bilder pro Video

# Hauptordner für Trainings- und Testdaten erstellen, falls sie nicht existieren
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Dictionary zum Tracken der Frame-Nummern für jede Klasse
frame_counters = {}


# Funktion zur Bildextraktion
def extract_images(video_path, train_ratio=0.5):
    # Videoname analysieren
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    book_title, volume_number, _ = video_name.split('_')
    class_name = f"{book_title}_{volume_number}"

    # Unterordner für die Klasse erstellen
    train_class_folder = os.path.join(train_folder, class_name)
    test_class_folder = os.path.join(test_folder, class_name)
    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(test_class_folder, exist_ok=True)

    # Frame-Zähler für die Klasse initialisieren, falls noch nicht vorhanden
    if class_name not in frame_counters:
        frame_counters[class_name] = 0

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Auswahl der Frames: jedes 10. Frame
    selected_frames = list(range(0, frame_count, frame_skip))

    # Auf maximale Anzahl begrenzen, falls mehr ausgewählt wurden
    if len(selected_frames) > max_frames_per_video:
        selected_frames = random.sample(selected_frames, max_frames_per_video)

    # Aufteilen in Trainings- und Testdaten
    split_index = int(len(selected_frames) * train_ratio)
    train_indices = selected_frames[:split_index]
    test_indices = selected_frames[split_index:]

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Frame nur speichern, wenn es in train_indices oder test_indices ist
        if current_frame in train_indices:
            save_folder = train_class_folder
        elif current_frame in test_indices:
            save_folder = test_class_folder
        else:
            current_frame += 1
            continue

        # Bildname mit globalem Frame-Zähler erstellen
        image_filename = f"{class_name}_frame{frame_counters[class_name]}.jpg"
        image_path = os.path.join(save_folder, image_filename)

        # Bild speichern
        cv2.imwrite(image_path, frame)
        print(f"Bild gespeichert: {image_path}")

        # Frame-Zähler für die Klasse erhöhen
        frame_counters[class_name] += 1
        current_frame += 1

    cap.release()


# Videos durchgehen und Bilder extrahieren
for video_file in os.listdir(video_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_folder, video_file)
        extract_images(video_path)
