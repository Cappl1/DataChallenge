import os
import csv
import pandas as pd
import shutil
from PIL import Image
import pickle
import random
import distutils.dir_util
import re


def copy_data_folder(src, dst):
    """
    Copies a folder from a source path to a destination path.
    If the destination folder exists, it will be overwritten.

    Parameters:
    - src (str): The path of the source folder.
    - dst (str): The path of the destination folder.

    Returns:
    None
    """
    # If the destination folder exists, remove it along with all its contents
    if os.path.exists(dst):
        shutil.rmtree(dst)
    

    try:
        shutil.copytree(src, dst)
        print(f'Successfully copied folder from {src} to {dst}')
    except Exception as e:
        print(f'An error occurred while copying the folder. Details: {str(e)}')

def get_filenames(directory):
    """Get every filename in a given directory"""
    subdirs = []
    filenames = []
    for root, _, files in os.walk(directory):
        for file in files:
            filenames.append(os.path.join(root, file))
            subdirs.append(os.path.basename(root))
    return filenames, subdirs


def extract(filenames, subdirs):
    """Extracts directory and filename information from the given file paths.

    Args:
        filenames (list): List of file paths.
        subdirs (list): List of subdirectories.

    Returns:
        list: List of tuples, each containing the directory, filename, and subdirectory of a file.
    """
    info = []

    for filename, subdirs in zip(filenames, subdirs):
        class_folder, image_name = os.path.split(filename)
        info.append((class_folder, image_name, subdirs))

    return info


def create_csv_dataset(directory, output_file):
    """Creates a CSV file containing information about files in a given directory.

    Args:
        directory (str): Path to the directory containing the files.
        output_file (str): Path to the output CSV file.
    """
    filenames, subdirs = get_filenames(directory)

    # Extract infos
    info = extract(filenames, subdirs)

    # Write the dataset to a csv file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "image_name", "class"])
        writer.writerows(info)
        
def drop_empty_classes(image_path, csv, threshold):
    """Deletes subdirectories in the given directory that contain fewer than a certain number of files.

    Args:
        image_path (str): Path to the directory containing the subdirectories.
        csv (str): Path to the CSV file containing information about the files.
        threshold (int): The minimum number of files a subdirectory should contain. If a subdirectory contains fewer 
                         files than this number, it will be deleted.
    """
    df = pd.read_csv(csv, delimiter=',', skiprows=0, low_memory=False, encoding='iso-8859-1')

    class_counts = df["class"].value_counts()

    for class_name, class_count in class_counts.items():
        if class_count < threshold * 2:
            subdirectory_path = os.path.join(image_path, str(class_name))

            if os.path.exists(subdirectory_path):
                shutil.rmtree(subdirectory_path)

    return

def merge_images(csv_file, output_dir):
    """
    Merges the front and back images of coins into a single image.
    
    Args:
        csv_file (str): Path to the CSV file containing information about the coin images.
        output_dir (str): Directory where the merged images will be stored.
    """
    # Define a function to merge the front and back side of the coin
    def merge(front_img_path, back_img_path):
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

    # Load CSV file
    df = pd.read_csv(csv_file, delimiter=',', skiprows=0, low_memory=False, encoding='iso-8859-1')

    # Create a new directory to store the merged images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through the CSV file and process each pair of images
    for index, row in df.iterrows():
        file_path = row['filename']
        filename = row['image_name']
        class_name = str(row['class'])  # Assuming the class column name is 'class'

        if 'obv' in filename:
            front_img_path = os.path.join(file_path, filename)
            back_img_path = os.path.join(file_path, filename.replace('obv', 'rev'))

            if os.path.exists(front_img_path) and os.path.exists(back_img_path):
                merged_img = merge(front_img_path, back_img_path)
                
                # Create a subdirectory for this class if it does not exist
                class_dir = os.path.join(output_dir, class_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                # Save the merged image in the respective class directory
                merged_filename = filename.replace('obv', 'merged')
                merged_img_path = os.path.join(class_dir, merged_filename)
                merged_img.save(merged_img_path)
                
                
def add_class_index(csv_file):
    """
    Adds an index to each unique class in a CSV file and saves the mapping to a pickle file.
    
    Args:
        csv_file (str): Path to the CSV file.
    """
    # Load CSV file
    df = pd.read_csv(csv_file, delimiter=',', skiprows=0, low_memory=False, encoding='iso-8859-1')

    # Get unique classes and assign them an index
    unique_classes = df["class"].unique()
    print("Number of Classes with samples", len(unique_classes))
    class_map = { geo: i for i, geo in enumerate(unique_classes)}

    # Add a column to the dataframe with the index of the class
    df["class_index"] = df.apply(lambda x: class_map[x["class"]], axis=1)

    # Save the modified dataframe to a new CSV file
    df[["filename", "image_name","class", "class_index"]].to_csv(csv_file, index=False)

    # Create reverse mapping and save it as a Python dict using pickle
    reverse_map = {v: k for k, v in class_map.items()}
    with open('class_index_map.pkl', 'wb') as fp:
        pickle.dump(reverse_map, fp)

    return

def split_dataset(dataset_path, train_path, validation_path, validation_percentage, seed=123):
    """
    Splits a dataset into a training set and a validation set.
    
    Args:
        dataset_path (str): Path to the dataset.
        train_path (str): Directory where the training set will be stored.
        validation_path (str): Directory where the validation set will be stored.
        validation_percentage (float): The percentage of the dataset to be used as the validation set.
        seed (int, optional): Seed for the random number generator.
    """
    # Set the seed for the random function
    if seed:
        random.seed(seed)

    # Clone the original dataset folder
    distutils.dir_util.copy_tree(dataset_path, train_path)

    # Walk through each subfolder in the cloned dataset folder
    for root, dirs, files in os.walk(train_path):

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
            dst = os.path.join(validation_path, os.path.relpath(root, train_path), file)
            shutil.move(src, dst)
            
            
def filenames_to_dataframe(directory):
    """Get every filename in a given directory"""
    filenames = []
    for root, _, files in os.walk(directory):
        for file in files:
            filenames.append(os.path.join(root, file))
            
    info = []
    for filename in filenames:
        class_folder, image_name = os.path.split(filename)
        info.append(image_name)
    
    return pd.DataFrame({'image_name': info})

def val_train_csv(all_images, train_or_test, output_dir, output_csv_file_name):
    """
    Generates a CSV file from a list of images, filtering out those that are not in a specified directory.
    
    Args:
        all_images (str): Path to the CSV file containing the full list of images.
        train_or_test (str): Directory containing the images to include in the output CSV.
        output_dir (str): Directory where the output CSV file will be stored.
        output_csv_file_name (str): Name of the output CSV file.
    """
    df_filenames = filenames_to_dataframe(train_or_test)

    # Load the CSV file
    df_csv = pd.read_csv(all_images, delimiter=',', skiprows=0, low_memory=False)

    # Filter the CSV file to keep only the rows corresponding to existing files
    df_filtered = pd.merge(df_filenames, df_csv, on='image_name')

    # Save the filtered DataFrame to a CSV file
    df_filtered.to_csv(os.path.join(output_dir, output_csv_file_name), index=False)
    
    
def add_type_description_to_csv(coin_csv, description_csv, output_csv):
    """
    Adds a column with the description of each coin type to a CSV file.
    
    Args:
        coin_csv (str): Path to the CSV file with information about the coins.
        description_csv (str): Path to the CSV file with descriptions of the coin types.
        output_csv (str): Path to the output CSV file.
    """
    # Read the first CSV
    df1 = pd.read_csv(coin_csv)

    # Extract the id from image_name
    df1['id'] = df1['image_name'].apply(lambda x: re.search('cn_coin_(.*?)_', x).group(1) if re.search('CN_type_(.*?)_', x) else None)
    
    # Read the second CSV as a single column
    df2 = pd.read_csv(description_csv, header=None, names=['combined'])

    # Split the combined column on the first comma
    df2[['id', 'description']] = df2['combined'].str.split(',', 1, expand=True)
    df2['id'] = df2['id'].str.strip('"')
    df2['description'] = df2['description'].str.strip('"')
    
    # Merge the two dataframes on id
    merged_df = pd.merge(df1, df2, on='id', how='inner')

    # Select and reorder columns
    merged_df = merged_df[['filename', 'image_name', 'class', 'class_index', 'description']]

    # Save to a new CSV
    merged_df.to_csv(output_csv, index=False)
    
    
def sanity_check(train_file_path, val_file_path, thresh):
    """
    Checks whether the training and validation datasets have the same number of unique class indices.

    Parameters:
    - train_file_path (str): The path to the training dataset CSV file.
    - val_file_path (str): The path to the validation dataset CSV file.
    - thresh (int): The threshold number of values per class.

    Returns:
    None
    """
    df = pd.read_csv(train_file_path)
    df2 = pd.read_csv(val_file_path)

    print(f"Your training data contains", len(df['class_index'].unique()) ,"distinct classes that have more than "+str(thresh)+" values and have a description available")
    print(f"Your validation data contains", len(df2['class_index'].unique()) ,"distinct classes that have more than "+str(thresh)+" values and have a description available")

    if len(df['class_index'].unique()) == len(df2['class_index'].unique()):
        print('You should be good to go')
    else: 
        print("You got a problem, validation and training classes do not match")
        
        
def clean_image_folders(csv_path: str, path_to_image_folder: str, count_threshold: int):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path, delimiter=',', skiprows=0, low_memory=False, encoding='iso-8859-1')
    
    # Calculate the value counts of the 'class' column
    class_counts = df["class"].value_counts()

    # Loop through the classes and their counts
    for class_name, class_count in class_counts.items():
        # If the count is less than the threshold
        if class_count < count_threshold:
            # Construct the path to the subdirectory for this class
            subdirectory_path = os.path.join(path_to_image_folder, class_name)

            # If the subdirectory exists, remove it
            if os.path.exists(subdirectory_path):
                shutil.rmtree(subdirectory_path)