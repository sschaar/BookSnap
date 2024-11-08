import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories for training and testing data
processed_train_dir = '../data/processed/images_with_points/processed_train_data/'  # Directory for processed training images
processed_test_dir = '../data/processed/images_with_points/processed_test_data/'    # Directory for processed test images

# Parameters for image size and batch size
img_height = 180
img_width = 180
batch_size = 32

# Step 1: Prepare data with ImageDataGenerator for data augmentation
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

# Step 2: Create train data generator
train_generator = datagen.flow_from_directory(
    processed_train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Step 3: Create test data generator
test_generator = datagen.flow_from_directory(
    processed_test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Step 4: Create the model
model = Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Dynamically set the number of classes
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Display model summary
model.summary()

# Step 7: Train the model
epochs = 50
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs
)

# Step 8: Display training accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Step 9: Save the model with versioning
model_dir = '../models/trained_model/'

# Check if the directory exists, and create it if necessary
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Get existing model versions
existing_versions = [f for f in os.listdir(model_dir) if f.endswith('.h5')]

# Determine the next version number
version_numbers = []
for version in existing_versions:
    try:
        # Extract the version number from the filename (e.g., 'book_cover_classification_model_v1.h5')
        version_number = int(version.split('_v')[-1].split('.')[0])
        version_numbers.append(version_number)
    except ValueError:
        continue

# Set the new version number as the next available version
new_version = max(version_numbers, default=0) + 1

# Create a new filename with the incremented version
model_save_path = os.path.join(model_dir, f'book_cover_classification_model_v{new_version}.h5')

# Save the model
model.save(model_save_path)
print(f"Model successfully saved: {model_save_path}")
