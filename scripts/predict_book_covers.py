import sys
import base64
from io import BytesIO

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model configuration
class_names = {0: "Berserk", 1: "Monster", 2: "Neon Genesis Evangelion"}
model_base_path = '/root/rately-files/models/'
model_version = '2'
model_path = f"{model_base_path}/book_cover_classification_model_v{model_version}.h5"

model = load_model(model_path)

img_height = 180
img_width = 180


def predict_image_class(base64_string):
    try:
        # Decode Base64 string
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = image.resize((img_width, img_height))

        # Prepare image for prediction
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        return class_names[predicted_index]
    except Exception as e:
        return f"Error processing image: {e}"


if __name__ == "__main__":
    base64_string = sys.stdin.read().strip()
    prediction = predict_image_class(base64_string)
    print(prediction)
