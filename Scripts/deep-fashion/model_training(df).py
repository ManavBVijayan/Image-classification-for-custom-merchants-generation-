import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('/Users/alkye/Desktop/Image_classification/deepfashion2_data.csv')

# Convert category_id to string
data['category_id'] = data['category_id'].astype(str)

# Define directories and file paths
images_dir = '/Users/alkye/Desktop/Image_classification/Dataset/deep-fashion/train'

# Check if all image paths exist
def check_image_paths(df, images_dir):
    missing_files = []
    for img_path in df['image_path']:
        full_path = os.path.join(images_dir, img_path)
        if not os.path.exists(full_path):
            missing_files.append(full_path)
    return missing_files

missing_files = check_image_paths(data, images_dir)
if missing_files:
    print(f"Found {len(missing_files)} missing image files.")
else:
    print("All image files are present.")

# Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Use 20% of data for validation
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=data,
    directory=images_dir,
    x_col='image_path',
    y_col='category_id',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',  # Use the training subset
    shuffle=True
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=data,
    directory=images_dir,
    x_col='image_path',
    y_col='category_id',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',  # Use the validation subset
    shuffle=False
)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')  # Use len(train_generator.class_indices) for number of classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save the model
model.save('clothing_classifier_model.h5')

print("Model training complete and saved as 'clothing_classifier_model.h5'.")