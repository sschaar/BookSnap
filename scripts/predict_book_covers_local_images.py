import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class_names = {0: "Berserk", 1: "Monster", 2: "Neon Genesis Evangelion"}

model_base_path = '../models/trained_model'
model_version = '2'
model_path = os.path.join(model_base_path, f'book_cover_classification_model_v{model_version}.h5')

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = tf.keras.models.load_model(model_path)
print(f"Loaded model version: {model_version} from {model_path}")

img_height = 180
img_width = 180

def predict_image_class(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_index]
    return predicted_class_name

inference_data_dir = '../data/inference_data'

for folder_name in os.listdir(inference_data_dir):
    folder_path = os.path.join(inference_data_dir, folder_name)
    if os.path.isdir(folder_path):
        print(f"\nPredictions for folder '{folder_name}':")
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            predicted_class_name = predict_image_class(image_path)
            print(f"Image: {image_name} - Predicted Class: {predicted_class_name}")
