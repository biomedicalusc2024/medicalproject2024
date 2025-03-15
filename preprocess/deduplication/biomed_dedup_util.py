# Same loading data, but here we have some different functions to deduplicate the dataset. 
import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Dataset Loading
def load_dataset(dataset_name, data_dir):
    """
    Load the dataset from the given path. It returns a dictionary with the file path as the key and the dataframe as the value for any file that is the given filetype in the given path.
    """
    if dataset_name == "bc5cdr":
        from ...dataLoader.relationExtraction.BC5CDR import getBC5CDR
        def reform_BC5CDR(ds):
            reformed = []
            for entity in ds:
                passage = entity["passages"]
                new_entity = {
                    "document_id": passage[0]["document_id"],
                    "text": passage[0]["text"] + " " + passage[1]["text"],
                    "relations": passage[0]["relations"],
                }
                reformed.append(new_entity)
            return pd.DataFrame(reformed)

        ds_raw = getBC5CDR(data_dir)
        ds = reform_BC5CDR(ds_raw[0]+ds_raw[1]+ds_raw[2])

    elif dataset_name == "BioNLI":
        from ...dataLoader.inference.BioNLI import getBioNLI

        ds_raw = getBioNLI(data_dir)
        ds = pd.DataFrame(ds_raw[0]+ds_raw[1]+ds_raw[2])

    elif dataset_name == "CORD19":
        from ...dataLoader.database.CORD19 import getCORD19

        ds_raw = getCORD19(data_dir, "fulltext")
        ds = pd.DataFrame(ds_raw)
        
    elif dataset_name == "hoc":
        from ...dataLoader.classification.HoC import getHoC

        ds_raw = getHoC(data_dir)
        ds = pd.DataFrame(ds_raw)
        
    elif dataset_name == "SourceData":
        from ...dataLoader.namedEntityRecognition.SourceData import getSourceData

        ds_raw = getSourceData(data_dir)
        ds = pd.DataFrame(ds_raw[0]+ds_raw[1]+ds_raw[2])
        
    return ds


def get_embeddings(texts, model, batch_size = 64):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc = "Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            embeddings.extend(model.encode(texts = batch_texts)['text_embeddings'])
    return np.array(embeddings)


def compute_similarity_chunked(embeddings, threshold=0.9, chunk_size=8000):
    """
    Compute cosine similarity in chunks to reduce memory usage.
    """
    n = len(embeddings)
    to_remove = set()
    for i in tqdm(range(0, n, chunk_size), desc= "Calcuating Similarity"):
        # Get the current chunk
        chunk_embeddings = embeddings[i:i + chunk_size]

        # Compute cosine similarity for the current chunk against all embeddings
        similarity_matrix = cosine_similarity(chunk_embeddings, embeddings)

        # Iterate through the chunk rows to find high-similarity indices
        for row_idx, similarities in enumerate(similarity_matrix):
            actual_idx = i + row_idx  # Map back to the original index
            if actual_idx in to_remove:
                continue

            similar_indices = np.where(similarities > threshold)[0]
            similar_indices = [idx for idx in similar_indices if idx > actual_idx]  # Avoid duplicates
            to_remove.update(similar_indices)

    return to_remove


def compute_similarity_between_datasets_chunked(embeddings1, embeddings2, threshold=0.9, chunk_size1=8000, chunk_size2=8000):
    """
    Compute cosine similarity between two datasets in chunks to reduce memory usage.
    Removes entries from embeddings1 based on high similarity with embeddings2.
    """
    to_remove = set()
    n1, n2 = len(embeddings1), len(embeddings2)

    for i in tqdm(range(0, n1, chunk_size1), desc="Processing dataset1 in chunks"):
        # Get a chunk from embeddings1
        chunk_embeddings1 = embeddings1[i:i + chunk_size1]

        for j in range(0, n2, chunk_size2):
            # Get a chunk from embeddings2
            chunk_embeddings2 = embeddings2[j:j + chunk_size2]

            # Compute cosine similarity for the two chunks
            similarity_matrix = cosine_similarity(chunk_embeddings1, chunk_embeddings2)

            # Check rows in chunk_embeddings1 with high similarity to chunk_embeddings2
            for row_idx, similarities in enumerate(similarity_matrix):
                actual_idx = i + row_idx  # Map back to the original index in embeddings1
                if actual_idx in to_remove:
                    continue
                if np.any(similarities > threshold):
                    to_remove.add(actual_idx)

    return to_remove


def deduplicate_within_dataset(dataset, columns, model, threshold=0.9):
    # joins the columns in the dataset
    texts = list(dataset[columns].apply(lambda x: " ".join(x.values.astype(str)), axis=1))
    embeddings = get_embeddings(texts, model)
    to_remove = compute_similarity_chunked(embeddings, threshold=threshold)
    number_removed = len(to_remove)
    droped_dataset = dataset.drop(to_remove)
    droped_dataset = droped_dataset.reset_index(drop= True)
    return droped_dataset, number_removed


def deduplicate_between_datasets(new_dataset, columns, model, old_embeddings, threshold=0.9):
    texts1 = list(new_dataset[columns].apply(lambda x: " ".join(x.values.astype(str)), axis=1))
    embeddings1 = get_embeddings(texts1, model)
    old_embeddings_list = []
    for embed in old_embeddings:
        old_embeddings_list.extend(embed)
    to_remove = compute_similarity_between_datasets_chunked(embeddings1, old_embeddings_list, threshold=threshold)
    number_removed = len(to_remove)
    droped_dataset = new_dataset.drop(to_remove)
    droped_dataset = droped_dataset.reset_index(drop= True)
    return droped_dataset, number_removed


def calculate_and_save_embeddings(dataset, dataset_name, model, column_names, save_dir="embeddings_cache", batch_size=128):
    """
    Compute and save embeddings for a QA dataset.

    Args:
        dataset (pd.DataFrame): Dataset containing "question" and "answer" columns.
        dataset_name (str): Name of the dataset for unique file identification.
        save_dir (str): Directory where embeddings will be saved.
        batch_size (int): Batch size for generating embeddings.

    Returns:
        dict: A dictionary containing question and answer embeddings.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # File paths for embeddings
    embedding_file = os.path.join(save_dir, f"{dataset_name}_embeddings.pkl")

    # Check if embeddings already exist
    if os.path.exists(embedding_file):
        print(f"Loading cached embeddings for {dataset_name}...")
        with open(embedding_file, "rb") as qf:
            embeddings = pickle.load(qf)
    else:
        with torch.no_grad():
            print(f"Generating embeddings for {dataset_name}...")
            if column_names == "all":
                texts = [' '.join(str(element) for element in row) for row in dataset.values] 
            else:
                texts = [' '.join(str(element) for element in row) for row in dataset[column_names].values] 
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Text Embeddings"):
                batch_texts = texts[i:i + batch_size]
                embeddings.extend(model.encode(texts=batch_texts)["text_embeddings"])
            embeddings = np.array(embeddings)

            # Save question embeddings
            with open(embedding_file, "wb") as qf:
                pickle.dump(embeddings, qf)
            print(f"Saved embeddings for {dataset_name}.")

        return embeddings