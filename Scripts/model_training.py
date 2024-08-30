import pandas as pd #type:ignore
import numpy as np #type:ignore
import matplotlib.pyplot as plt #type:ignore
from tensorflow.keras import layers, models, callbacks, optimizers #type:ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore

# Load data from CSV files
train_data = pd.read_csv('/Users/alkye/Desktop/Image_classification/Dataset/fashion-mnist/fashion-mnist_train.csv')
test_data = pd.read_csv('/Users/alkye/Desktop/Image_classification/Dataset/fashion-mnist/fashion-mnist_test.csv')

# Prepare the data
train_labels = train_data['label'].values
train_images = train_data.drop(columns=['label']).values.reshape(-1, 28, 28, 1) / 255.0

test_labels = test_data['label'].values
test_images = test_data.drop(columns=['label']).values.reshape(-1, 28, 28, 1) / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(train_images)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Added Dropout to prevent overfitting
    layers.Dense(10, activation='softmax')
])

# Compile the model with a learning rate scheduler
initial_learning_rate = 0.001
lr_schedule = callbacks.LearningRateScheduler(
    lambda epoch: initial_learning_rate * 10**(epoch / 10))

model.compile(optimizer=optimizers.Adam(learning_rate=initial_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with data augmentation
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    epochs=30,
                    validation_data=(test_images, test_labels),
                    callbacks=[early_stopping, lr_schedule])

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
model.save('fashion_mnist_model_augmented.h5')