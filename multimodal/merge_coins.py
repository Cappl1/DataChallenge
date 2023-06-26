import os
import pandas as pd
from PIL import Image

# Load CSV file
df = pd.read_csv(r"F:\Users\basti\Documents\Goethe Uni\Data Challange\datasettypes1.csv", delimiter=',', skiprows=0, low_memory=False, encoding='iso-8859-1')

# Define a function to merge the front and back side of the coin
def merge_images(front_img_path, back_img_path):
    # Open images and resize to half the final height
    front_img = Image.open(front_img_path)
    front_img = front_img.resize((149,149))
    
    back_img = Image.open(back_img_path)
    back_img = back_img.resize((149,149))

    # Create a new image with black background
    merged_img = Image.new('RGB', (299, 299), (0, 0, 0))

    # Paste the images. For the backside image, it starts halfway down the final image.
    merged_img.paste(front_img, (0,75))
    merged_img.paste(back_img, (150,75))

    return merged_img

# Create a new directory to store the merged images
if not os.path.exists('merged_coins'):
    os.makedirs('merged_coins')

# Loop through the CSV file and process each pair of images
for index, row in df.iterrows():
    file_path = row['filename']
    filename = row['image_name']
    class_name = str(row['class'])  # Assuming the class column name is 'class'

    if 'obv' in filename:
        front_img_path = os.path.join(file_path, filename)
        back_img_path = os.path.join(file_path, filename.replace('obv', 'rev'))

        if os.path.exists(front_img_path) and os.path.exists(back_img_path):
            merged_img = merge_images(front_img_path, back_img_path)
            
            # Create a subdirectory for this class if it does not exist
            class_dir = os.path.join('merged_coins', class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Save the merged image in the respective class directory
            merged_filename = filename.replace('obv', 'merged')
            merged_img_path = os.path.join(class_dir, merged_filename)
            merged_img.save(merged_img_path)