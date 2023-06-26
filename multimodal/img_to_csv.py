import os
import csv

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
    info = []

    for filename, subdirs in zip(filenames, subdirs):
        class_folder, image_name = os.path.split(filename)
        info.append((class_folder, image_name, subdirs))

    return info


def create_csv_dataset(directory, output_file):
    # Get the filenames
    filenames, subdirs = get_filenames(directory)

    # Extract infos
    info = extract(filenames, subdirs)

    # Write the dataset to a csv file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "image_name", "class"])
        writer.writerows(info)


if __name__ == "__main__":
    IMAGE_DIR = r"F:\Users\basti\Documents\Goethe Uni\Data Challange\CN_dataset_04_23\merged_coins"
    OUTPUT_FILE = "datasettypes2.csv"

    # Create the dataset CSV file
    create_csv_dataset(IMAGE_DIR, OUTPUT_FILE)