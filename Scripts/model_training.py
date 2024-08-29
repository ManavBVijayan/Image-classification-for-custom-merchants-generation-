import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Load data from CSV files
train_data = pd.read_csv('/Users/alkye/Desktop/Image_classification/Dataset/fashion-mnist_train.csv')
test_data = pd.read_csv('/Users/alkye/Desktop/Image_classification/Dataset/fashion-mnist_test.csv')

# Prepare the data
train_labels = train_data['label'].values
train_images = train_data.drop(columns=['label']).values.reshape(-1, 28, 28, 1) / 255.0

test_labels = test_data['label'].values
test_images = test_data.drop(columns=['label']).values.reshape(-1, 28, 28, 1) / 255.0

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Plot accuracy and loss over epochs
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Save the trained model
model.save('fashion_mnist_model.h5')