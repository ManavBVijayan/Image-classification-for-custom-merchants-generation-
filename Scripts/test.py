import os
import pandas as pd
# Assuming `train_data` is your DataFrame and `images_dir` is the directory with images
images_dir = os.path.join('/Users/alkye/Desktop/Image_classification/Dataset/deep-fashion', 'train')
data = pd.read_csv('/Users/alkye/Desktop/Image_classification/deepfashion2_data.csv')

# Print some paths and check if they exist
for img_path in data['image_path'].head(10):
    full_path = os.path.join(images_dir, img_path)
    print(f"Checking: {full_path}")
    if not os.path.exists(full_path):
        print(f"Missing file: {full_path}")