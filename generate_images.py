import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# Paths to folders
input_folder_authentic = 'dataset/authentic'
input_folder_forged = 'dataset/forged'
output_folder_authentic = 'augmented/authentic'
output_folder_forged = 'augmented/forged'

# Create output directories if they don't exist
os.makedirs(output_folder_authentic, exist_ok=True)
os.makedirs(output_folder_forged, exist_ok=True)

# Define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

def augment_images(input_folder, output_folder, target_count):
    # List all images in the input folder
    images = os.listdir(input_folder)
    current_count = len(images)
    augment_per_image = (target_count - current_count) // current_count + 1  # Augmentations per image
    
    # Process each image
    for img_file in images:
        img_path = os.path.join(input_folder, img_file)
        img = load_img(img_path, color_mode='grayscale')  # Load image as grayscale
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Generate augmented images
        i = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_folder, save_prefix='aug', save_format='png'):
            i += 1
            if i >= augment_per_image:  # Stop when enough augmentations are created
                break
    
    # Ensure the total number of images is exactly the target count
    final_images = os.listdir(output_folder)
    while len(final_images) < target_count:
        for img_file in images:
            if len(final_images) >= target_count:
                break
            img_path = os.path.join(input_folder, img_file)
            img = load_img(img_path, color_mode='grayscale')
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_folder, save_prefix='aug', save_format='png'):
                break
        final_images = os.listdir(output_folder)

# Generate augmented images
augment_images(input_folder_authentic, output_folder_authentic, target_count=200)
augment_images(input_folder_forged, output_folder_forged, target_count=200)

print("Augmentation complete! Augmented images saved in 'augmented' folder.")
