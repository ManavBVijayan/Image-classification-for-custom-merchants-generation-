from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os

# Define constants
TARGET_SIZE = (28, 28)  # Size used for Fashion MNIST
MODEL_PATH = '/Users/alkye/Desktop/Image_classification/fashion_mnist_model.h5'  # Path to your trained model
CLASS_LABELS = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def preprocess_image(image_path, target_size=TARGET_SIZE):
    # Load and resize the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape((1, target_size[0], target_size[1], 1))  # Reshape for model input
    return image_array, image

def load_cnn_model(model_path=MODEL_PATH):
    return load_model(model_path)

def predict_clothing(image_array, model):
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

def get_clothing_label(predicted_class):
    return CLASS_LABELS[predicted_class[0]]

def visualize_image(original_image, preprocessed_image, label):
    # Plot original image
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')
    
    # Plot preprocessed image
    plt.subplot(1, 2, 2)
    plt.title('Preprocessed Image')
    plt.imshow(preprocessed_image.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.suptitle(f'Predicted: {label}')
    plt.show()

def classify_clothing(image_path):
    image_array, original_image = preprocess_image(image_path)
    model = load_cnn_model()
    predicted_class = predict_clothing(image_array, model)
    clothing_label = get_clothing_label(predicted_class)
    
    # Visualize the images and prediction
    visualize_image(original_image, image_array[0], clothing_label)
    
    return clothing_label

if __name__ == "__main__":
    # Example usage
    image_path = '/Users/alkye/Desktop/Image_classification/trousers.jpg'  # Replace with the path to your image
    if os.path.exists(image_path):
        label = classify_clothing(image_path)
        print(f'The clothing item in the image is a {label}.')
    else:
        print("Image file not found.")