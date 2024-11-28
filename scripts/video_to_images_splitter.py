import cv2
import os
import random

video_folder = ('./data/raw/video')
train_folder = './data/processed/images/train_data'
test_folder = './data/processed/images/test_data'

frame_skip = 10
max_frames_per_video = 500

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

frame_counters = {}

def extract_images(video_path, train_ratio=0.5):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    book_title, volume_number, _ = video_name.split('_')
    class_name = f"{book_title}_{volume_number}"

    train_class_folder = os.path.join(train_folder, class_name)
    test_class_folder = os.path.join(test_folder, class_name)
    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(test_class_folder, exist_ok=True)

    if class_name not in frame_counters:
        frame_counters[class_name] = 0

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    selected_frames = list(range(0, frame_count, frame_skip))

    if len(selected_frames) > max_frames_per_video:
        selected_frames = random.sample(selected_frames, max_frames_per_video)

    split_index = int(len(selected_frames) * train_ratio)
    train_indices = selected_frames[:split_index]
    test_indices = selected_frames[split_index:]

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in train_indices:
            save_folder = train_class_folder
        elif current_frame in test_indices:
            save_folder = test_class_folder
        else:
            current_frame += 1
            continue

        image_filename = f"{class_name}_frame{frame_counters[class_name]}.jpg"
        image_path = os.path.join(save_folder, image_filename)

        cv2.imwrite(image_path, frame)
        print(f"Bild gespeichert: {image_path}")

        frame_counters[class_name] += 1
        current_frame += 1

    cap.release()


for video_file in os.listdir(video_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_folder, video_file)
        extract_images(video_path)
