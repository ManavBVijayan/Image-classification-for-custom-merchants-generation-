import tensorflow as tf
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first image from the training set
plt.figure()
plt.imshow(train_images[7], cmap='gray')
plt.title(f'Label: {train_labels[0]}')
plt.show()