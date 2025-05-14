import os
import shutil
import random

# Configuration
data_dir_images = 'E:/transformed_dataset/images/'  # Directory containing all images and annotations
data_dir_labels = 'E:/transformed_dataset/labels/'  # Directory containing all images and annotations

output_dir = 'C:/Users/CanAliYarman/Documents/MMrot/ETDII'  # Base output directory
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create directories
os.makedirs(os.path.join(output_dir, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val/labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test/labels'), exist_ok=True)

# Get list of all image files and corresponding annotation files
image_files = [f for f in os.listdir(data_dir_images) if f.endswith('.png')]
annotation_files = [f for f in os.listdir(data_dir_labels) if f.endswith('.txt')]
print(annotation_files)
# Shuffle and split the dataset
combined = list(zip(image_files, annotation_files))
random.shuffle(combined)
num_train = int(len(combined) * train_ratio)
num_val = int(len(combined) * val_ratio)
num_test = len(combined) - num_train - num_val

train_set = combined[:num_train]
val_set = combined[num_train:num_train + num_val]
test_set = combined[num_train + num_val:]

# Helper function to move files
def move_files(file_set, subset):
    for img_file, ann_file in file_set:
        shutil.move(os.path.join(data_dir_images, img_file), os.path.join(output_dir, f'{subset}/images/', img_file))
        shutil.move(os.path.join(data_dir_labels, ann_file), os.path.join(output_dir, f'{subset}/labels/', ann_file))

# Move files to respective directories
move_files(train_set, 'train')
move_files(val_set, 'val')
move_files(test_set, 'test')

print(f"Dataset split into train, val, and test sets.\n"
      f"Train set: {len(train_set)} samples\n"
      f"Validation set: {len(val_set)} samples\n"
      f"Test set: {len(test_set)} samples")