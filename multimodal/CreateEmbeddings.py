import openai
import pandas as pd
import csv
import numpy as np

openai.api_key = 

# Read the CSV file using pandas
df = pd.read_csv('val1.csv')

# Initialize an empty list to store the embeddings
embeddings = []
print(len(df['description']))
# Loop through each row in the dataframe
for text in df['description']:
    # Generate the embedding for the text
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    # Retrieve the embedding from the response
    embedding = response['data'][0]['embedding']
    # Append the embedding to the list
    embeddings.append(embedding)
    


# Convert the list of embeddings to a NumPy array
embeddings = np.array(embeddings)

# Save the embeddings to a binary file using NumPy
np.save('embeddings_val.npy', embeddings)
