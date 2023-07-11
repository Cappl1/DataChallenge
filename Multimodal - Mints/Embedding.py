import openai
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential

@retry(wait=wait_random_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def create_embedding(text, model):
    # This is the function that might fail and needs to be retried
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response

def generate_embeddings(api_key, csv_path, model, save_path):
    openai.api_key = api_key

    # Read the CSV file using pandas
    df = pd.read_csv(csv_path)

    # Initialize an empty list to store the embeddings
    embeddings = []

    # The description column
    description_column = "description"

    # Loop through each row in the dataframe
    for index, row in df.iterrows():
        if pd.isna(row[description_column]):  # Skip if the description is NaN
            continue
        try:
            # Generate the embedding for the text
            response = create_embedding(row[description_column], model)
            # Retrieve the embedding from the response
            embedding = response['data'][0]['embedding']
            # Append the embedding to the list
            if len(embedding)!=1536:
                    
                print('problem in row', index)
                problems += 1
                embedding = 1536 * [0]
            embeddings.append(embedding)
        except Exception as e:
            print(f"Failed to generate embedding for {row[description_column]} due to {str(e)}")

    # Convert the list of embeddings to a NumPy array
    embeddings = np.array(embeddings)

    # Save the embeddings to a binary file using NumPy
    np.save(save_path, embeddings)

# usage
generate_embeddings('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', 'unique_labels.csv', 'text-embedding-ada-002', 'embeddings_val_original.npy')
generate_embeddings('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', 'unique_labels.csv', 'text-embedding-ada-002', 'embeddings_train_original.npy')