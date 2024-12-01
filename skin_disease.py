# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Constants
IMAGE_SIZE = (96, 96)

# Load the dataset
data = pd.read_csv('H:\skin_disease_detection\data\HAM10000_metadata')  # Adjust path
image_dir = 'H:\skin_disease_detection\images'  # Adjust path

# Load images and labels
images, labels = [], []
for index, row in data.iterrows():
    image_path = os.path.join(image_dir, row['image_id'] + '.jpg')
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, IMAGE_SIZE)
        images.append(image)
        labels.append(row['dx'])  # Diagnosis

# Convert to numpy arrays and preprocess
X = np.array(images, dtype='float32') / 255.0
y = pd.get_dummies(labels).values  # One-hot encode labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)
datagen.fit(X_train)

# Define CNN model
def create_cnn_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

num_classes = y_train.shape[1]
model = create_cnn_model(num_classes)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

# Training
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=30,
                    validation_data=(X_test, y_test), callbacks=[early_stopping, lr_scheduler])

# Save the model
output_dir = os.path.abspath(os.path.join('H:\skin_disease_detection', 'output'))
os.makedirs(output_dir, exist_ok=True)
model.save(os.path.join(output_dir, 'best_model.keras'))

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
plt.show()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
