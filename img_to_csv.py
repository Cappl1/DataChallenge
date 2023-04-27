import csv
import os


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


def main():
    IMAGE_DIR = r"F:\Users\basti\Documents\Goethe Uni\Data Challange\CN_dataset_04_23\dataset_mints"

    # Get the filenames
    filenames, subdirs = get_filenames(IMAGE_DIR)
    
    # Extract infos
    info = extract(filenames, subdirs)

    # Write the dataset to a csv file
    with open("dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "image_name","class"])
        writer.writerows(info)


if __name__ == "__main__":
    main()