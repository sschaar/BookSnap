import os
import cv2
import numpy as np
import shutil

# Verzeichnisse für Eingabe und Ausgabe
train_data_dir = '../data/processed/images/train_data/'  # Ordner für Trainingsbilder
test_data_dir = '../data/processed/images/test_data/'  # Ordner für Testbilder
processed_train_dir = '../data/processed/images_with_points/processed_train_data/'  # Ausgabeordner für verarbeitete Trainingsbilder
processed_test_dir = '../data/processed/images_with_points/processed_test_data/'  # Ausgabeordner für verarbeitete Testbilder

# Leere die Ausgabeordner, falls sie existieren
for output_dir in [processed_train_dir, processed_test_dir]:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

# Funktion zur Erkennung der Eckpunkte und Zeichnen von Linien über das gesamte Bild
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

# Funktion zum Verarbeiten eines Ordners mit Bildern
def process_images(input_dir, output_dir):
    for book_folder in os.listdir(input_dir):
        book_folder_path = os.path.join(input_dir, book_folder)

        if os.path.isdir(book_folder_path):
            print(f"Lade und verarbeite Bilder für das Buch: {book_folder}")
            book_output_dir = os.path.join(output_dir, book_folder)
            os.makedirs(book_output_dir, exist_ok=True)

            for idx, image_name in enumerate(os.listdir(book_folder_path)):
                image_path = os.path.join(book_folder_path, image_name)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Bild {image_name} konnte nicht geladen werden.")
                    continue

                # Bild verarbeiten und Eckpunkte markieren
                image_with_points = detect_corners(image)

                # Speichere das verarbeitete Bild
                output_image_path = os.path.join(book_output_dir, f"{book_folder}_{idx}_points.jpg")
                cv2.imwrite(output_image_path, image_with_points)
                print(f"Gespeichertes Bild: {output_image_path}")

# Verarbeite die Trainings- und Testbilder
process_images(train_data_dir, processed_train_dir)
process_images(test_data_dir, processed_test_dir)
