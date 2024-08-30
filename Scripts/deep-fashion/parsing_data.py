import os
import json
import pandas as pd  # type: ignore

# Define paths to the images and annotation directories
train_images_dir = '/Users/alkye/Desktop/Image_classification/Dataset/deep-fashion/train/image'
train_annos_dir = '/Users/alkye/Desktop/Image_classification/Dataset/deep-fashion/train/annos'

data = []

# Loop through annotation files
for anno_file in os.listdir(train_annos_dir):
    if anno_file.endswith('.json'):
        with open(os.path.join(train_annos_dir, anno_file)) as f:
            anno_data = json.load(f)

        # Extract information
        image_id = anno_file.split('.')[0]
        image_name = image_id + '.jpg'  # Update extension to .jpg
        
        # Only save folder name and filename
        folder_name = os.path.basename(train_images_dir)
        image_path = os.path.join(folder_name, image_name)
        
        # Extract data for item1 and item2
        for item_key in ['item1', 'item2']:
            item_data = anno_data.get(item_key, {})
            
            category_id = item_data.get('category_id', None)
            category_name = item_data.get('category_name', None)
            bounding_box = item_data.get('bounding_box', None)
            
            # Append data
            data.append([image_path, category_id, category_name, bounding_box])

# Create a DataFrame
df = pd.DataFrame(data, columns=['image_path', 'category_id', 'category_name', 'bounding_box'])

# Save to CSV
df.to_csv('deepfashion2_data.csv', index=False)

print("Data parsing complete. CSV saved as 'deepfashion2_data.csv'.")