import pandas as pd #type:ignore
import numpy as np #type:ignore
import matplotlib.pyplot as plt #type:ignore
from tensorflow.keras import layers, models, callbacks, optimizers #type:ignore
from tensorflow.keras.applications import ResNet50 #type:ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
from tensorflow.keras.applications import MobileNetV2 #type:ignore

# Load data from CSV files
train_data = pd.read_csv('/Users/alkye/Desktop/Image_classification/Dataset/fashion-mnist/fashion-mnist_train.csv')
test_data = pd.read_csv('/Users/alkye/Desktop/Image_classification/Dataset/fashion-mnist/fashion-mnist_test.csv')

# Prepare the data
train_labels = train_data['label'].values
train_images = train_data.drop(columns=['label']).values.reshape(-1, 28, 28, 1)

test_labels = test_data['label'].values
test_images = test_data.drop(columns=['label']).values.reshape(-1, 28, 28, 1)

# Resize images to 224x224 to match the input size expected by ResNet50
train_images = np.repeat(train_images, 3, axis=-1)  # Convert to 3 channels
train_images = np.array([np.resize(img, (224, 224, 3)) for img in train_images])
test_images = np.repeat(test_images, 3, axis=-1)  # Convert to 3 channels
test_images = np.array([np.resize(img, (224, 224, 3)) for img in test_images])

# Split training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

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

# Load the pre-trained ResNet50 model without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
print("It's done")

# Add custom layers on top of the base model
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)  # Dropout to prevent overfitting
predictions = layers.Dense(10, activation='softmax')(x)  # 10 classes for Fashion MNIST

# Define the full model
model = models.Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping and checkpoint callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model with data augmentation
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    epochs=30,
                    validation_data=(val_images, val_labels),
                    callbacks=[early_stopping, checkpoint])

# Fine-tuning: Unfreeze some layers of the base model
for layer in base_model.layers[-10:]:  # Unfreeze last 10 layers for fine-tuning
    layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Continue training with fine-tuning
fine_tune_history = model.fit(datagen.flow(train_images, train_labels, batch_size=16),
                              epochs=10,
                              validation_data=(val_images, val_labels),
                              callbacks=[early_stopping, checkpoint])

# Load the best model saved during training
model.load_weights('best_model.h5')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Plot accuracy and loss over epochs
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(fine_tune_history.history['accuracy'], label='Fine-Tuning Accuracy')
plt.plot(fine_tune_history.history['val_accuracy'], label='Fine-Tuning Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(fine_tune_history.history['loss'], label='Fine-Tuning Loss')
plt.plot(fine_tune_history.history['val_loss'], label='Fine-Tuning Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Save the final fine-tuned model
model.save('fashion_mnist_resnet50_finetuned.h5')