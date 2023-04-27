#clone orignal folder first 
import os
import pandas as pd
import shutil

df = pd.read_csv(r"F:\Users\basti\Documents\Goethe Uni\Data Challange\dataset.csv", delimiter=',', skiprows=0, low_memory=False, encoding='iso-8859-1')

class_counts = df["class"].value_counts()

class_counts_below_20 = class_counts[class_counts < 20].count()


path_to_image_folder = r"F:\Users\basti\Documents\Goethe Uni\Data Challange\CN_dataset_04_23\dataset_mints"

for class_name, class_count in class_counts.items():
    if class_count < 20:
        subdirectory_path = os.path.join(path_to_image_folder, class_name)
        
            
        if os.path.exists(subdirectory_path):
            shutil.rmtree(subdirectory_path)