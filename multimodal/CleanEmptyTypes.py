#clone orignal folder first 
import os
import pandas as pd
import shutil

df = pd.read_csv(r"F:\Users\basti\Documents\Goethe Uni\Data Challange\datasettypes.csv", delimiter=',', skiprows=0, low_memory=False, encoding='iso-8859-1')

class_counts = df["class"].value_counts()

class_counts_below_40 = class_counts[class_counts < 40].count()


path_to_image_folder = r"F:\Users\basti\Documents\Goethe Uni\Data Challange\CN_dataset_04_23\dataset_coins"

for class_name, class_count in class_counts.items():
    if class_count < 40:
        subdirectory_path = os.path.join(path_to_image_folder, str(class_name))
        
            
        if os.path.exists(subdirectory_path):
            shutil.rmtree(subdirectory_path)