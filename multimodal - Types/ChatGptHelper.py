import pandas as pd
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
import numpy as np

def extract_unique_labels(file_path, new_file_path):
    """Reads a CSV file, extracts unique class indices, and saves a new CSV file with unique class indices.

    Parameters:
    - file_path (str): The path of the original CSV file.
    - new_file_path (str): The path where the new CSV file should be saved.

    Returns:
    None
    """
    df = pd.read_csv(file_path, delimiter=',', skiprows=0, low_memory=False, encoding='iso-8859-1')

    unique_class_indices = df['class_index'].unique()

    new_data = pd.DataFrame({'class_index': unique_class_indices})

    new_data = pd.merge(new_data, df.drop_duplicates(subset=['class_index']), on='class_index')

    column_order = ['filename', 'image_name', 'class', 'class_index', 'description']
    new_data = new_data[column_order]

    new_data.to_csv(new_file_path, index=False)
    
    
@retry(wait=wait_random_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def get_alternative_description(api_key, description):
    """Get an alternative description for a historical coin from GPT-4.

    Args:
        api_key (str): OpenAI API key.
        description (str): The original description of a historical coin.

    Returns:
        str: The alternative description of the historical coin.

    The function is designed to retry 5 times with exponential backoff if the request fails.
    """
    
    openai.api_key = api_key
    parameters = {  
        'model': 'gpt-4',   
        'messages': [
            {"role": "system", "content": "You are a helpful assistant."},   
            {"role": "user", "content": f'This is a description of a historical coin: "{description}". Please find me an alternative formulation of the same description. Please keep the description very similar.'}
        ]
    }
    response = openai.ChatCompletion.create(**parameters)
    return response['choices'][0]['message']['content'].strip()

def generate_alternative_descriptions(api_key, input_file, output_file, no_of_alternative_descriptions=10):
    """Generate alternative descriptions for a list of historical coins.

    Args:
        api_key (str): OpenAI API key.
        input_file (str): Path to the input CSV file containing descriptions of historical coins.
        output_file (str): Path to the output CSV file to store the original and alternative descriptions.
        no_of_alternative_descriptions (int, optional): Number of alternative descriptions to generate for each coin. Defaults to 10.

    This function reads the original descriptions from the input file, generates a specified number of alternative 
    descriptions using GPT-4 for each coin, and writes the original and alternative descriptions to the output file.
    """
    df = pd.read_csv(input_file)
    for index, row in df.iterrows():
        if index % 10 == 0:
            print(index ,"/", len(df), "generated")
        description = row['description']
        for i in range(no_of_alternative_descriptions):
            alternative_description = get_alternative_description(api_key, description)
            df.loc[index, f'description_{i+1}'] = alternative_description

    df.to_csv(output_file, index=False)
    
    
@retry(wait=wait_random_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def create_embedding(text, model):
    """Create an embedding for a given text using a specified model.

    Args:
        text (str): The text to generate an embedding for.
        model (str): The model to use for generating the embedding.

    Returns:
        dict: The response from the OpenAI API containing the generated embedding.

    This function might fail due to API errors, in which case it is designed to retry 5 times 
    with exponential backoff before stopping.
    """
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response

def generate_embeddings(api_key, csv_path, model, save_path, no_of_alternative_descriptions=10):
    """Generate embeddings for a list of texts using a specified model.

    Args:
        api_key (str): OpenAI API key.
        csv_path (str): Path to the CSV file containing the texts.
        model (str): The model to use for generating the embeddings.
        save_path (str): Path to the output file where the generated embeddings will be saved.
        no_of_alternative_descriptions (int, optional): Number of alternative descriptions to consider for each text. Defaults to 10.

    This function reads the texts from the CSV file, generates an embedding for each text using the specified model, 
    and writes the embeddings to the output file. If generating an embedding for a text fails, the function will log the error 
    and continue with the next text.
    """
    openai.api_key = api_key

    # Read the CSV file using pandas
    df = pd.read_csv(csv_path, delimiter=';', encoding='iso-8859-1')
    
    # Initialize an empty list to store the embeddings
    embeddings = []

    # Get the description columns
    description_columns = [f"description_{i}" for i in range(1, no_of_alternative_descriptions+1)]
    problems = 0
    # Loop through each row in the dataframe
    for index, row in df.iterrows():
        row_embeddings = []
        # Loop through each description column
        for col in description_columns:
            if pd.isna(row[col]):  # Skip if the description is NaN
                continue
            try:
                # Generate the embedding for the text
                response = create_embedding(row[col], model)
                # Retrieve the embedding from the response
                embedding = response['data'][0]['embedding']
                # Append the embedding to the row's list
                if len(embedding)!=1536:
                    
                    print('problem in row', index, 'and', col)
                    problems += 1
                    embedding = 1536 * [0]
                    
                row_embeddings.append(embedding)
            except Exception as e:
                print(f"Failed to generate embedding for {row[col]} due to {str(e)}")
                continue
        
        # Append the row's embeddings to the main list, only if there are any embeddings
        if row_embeddings:
            embeddings.append(row_embeddings)

    # Convert the list of embeddings to a NumPy array
    embeddings = np.array(embeddings)

    # Save the embeddings to a binary file using NumPy
    np.save(save_path, embeddings)
    
    print('Total embeding processes with ', problems ,'problems')
    
