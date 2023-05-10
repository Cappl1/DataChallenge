import os
import random
import shutil

def split_dataset(dataset_path, validation_path, validation_percentage, seed=None):
    # Set the seed for the random function
    if seed:
        random.seed(seed)

    # Walk through each subfolder in the dataset folder
    for root, dirs, files in os.walk(dataset_path):

        # Create a corresponding subfolder in the validation dataset folder
        for dir in dirs:
            validation_dir = os.path.join(validation_path, dir)
            os.makedirs(validation_dir, exist_ok=True)

        # Calculate the number of images to move to the validation dataset folder
        num_validation = int(len(files) * validation_percentage)

        # Randomly select the images to move
        validation_files = random.sample(files, num_validation)

        # Move the selected images to the validation dataset folder
        for file in validation_files:
            src = os.path.join(root, file)
            dst = os.path.join(validation_path, os.path.relpath(root, dataset_path), file)
            shutil.move(src, dst)

if __name__ == "__main__":
    # Set the path to your dataset folder
    dataset_path = r"F:\Users\basti\Documents\Goethe Uni\Data Challange\CN_dataset_04_23\dataset_mints_train"

    # Set the path to your validation dataset folder
    validation_path = r"F:\Users\basti\Documents\Goethe Uni\Data Challange\CN_dataset_04_23\dataset_mints_val"

    # Set the percentage of images you want in your validation set
    validation_percentage = 0.25

    # Set the seed for the random function (optional)
    seed = 123

    # Split the dataset into training and validation
    split_dataset(dataset_path, validation_path, validation_percentage, seed)