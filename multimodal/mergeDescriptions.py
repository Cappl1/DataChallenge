import pandas as pd
import re

def merge_csvs(csv1, csv2, output_csv):
    # Read the first CSV
    df1 = pd.read_csv(csv1)

    # Extract the id from image_name
    df1['id'] = df1['image_name'].apply(lambda x: re.search('CN_type_(.*?)_', x).group(1) if re.search('CN_type_(.*?)_', x) else None)
    
    # Read the second CSV as a single column
    df2 = pd.read_csv(csv2, header=None, names=['combined'])

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
    
    
def check_ids(csv1, csv2):
    # Read the first CSV
    df1 = pd.read_csv(csv1)
    df1['id'] = df1['image_name'].apply(lambda x: re.search('CN_type_(.*?)_', x).group(1) if re.search('CN_type_(.*?)_', x) else None)

    # Read the second CSV as a single column
    df2 = pd.read_csv(csv2, header=None, names=['combined'])
    df2[['id', 'description']] = df2['combined'].str.split(',', 1, expand=True)
    df2['id'] = df2['id'].str.strip('"')
    df2['description'] = df2['description'].str.strip('"')

    # Merge the two dataframes on id, but keep all records from df1 even if there's no match in df2
    merged_df = pd.merge(df1, df2, on='id', how='left')

    # Check for missing ids
    missing_ids = merged_df[merged_df['description'].isnull()]['id'].unique()

    # Print missing ids
    print(f"Missing ids: {missing_ids}")

if __name__ == '__main__':
    DIR = r"F:\Users\basti\Documents\Goethe Uni\Data Challange\multimodal"
    
    OUTPUT_FILE_train = "train1.csv"
    OUTPUT_FILE_val = "val1.csv"
    #training data
    check_ids(DIR + '\\train.csv', DIR + '\\CN_coin_descriptions.csv')
    
    #val data 
    check_ids(DIR + '\\val.csv', DIR + '\\CN_coin_descriptions.csv')
    
    #training data
    merge_csvs(DIR + '\\train.csv', DIR + '\\CN_coin_descriptions.csv', OUTPUT_FILE_train)
    
    #val data 
    merge_csvs(DIR + '\\val.csv', DIR + '\\CN_coin_descriptions.csv', OUTPUT_FILE_val)