# Utility functions for QA deduplication
# This file contains all the functions that are used in the QA deduplication notebook.

# imports
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
    Load the dataset from the given path. 
    """
    if dataset_name == "LiveQA":
        from ...dataLoader.questionAnswering.LiveQA_TREC_2017 import getLiveQA_TREC_2017
        def reform_LiveQA(ds):
            reformed = []
            columns = [
                'NIST_PARAPHRASE', 'NLM_SUMMARY', 'QUESTION_ID', 
                'ORIGINAL_QUESTION_SUBJECT', 'ORIGINAL_QUESTION_FILE', 
                'ANNOTATIONS_FOCUS', 'ANNOTATIONS_TYPE', 'ANNOTATIONS_KEYWORD'
            ]
            for entity in ds:
                new_entity = {q:entity[q] for q in columns}
                new_entity["question"] = entity["ORIGINAL_QUESTION_MESSAGE"]
                new_entity["answer"] = [a["ANSWER"] for a in entity["REFERENCE_ANSWERS"]]
                reformed.append(new_entity)
            df = pd.DataFrame(reformed)
            df["question"] = df["question"].fillna("").astype(str)
            df["answer"] = df["answer"].fillna("").astype(str)
            df = df[(df["question"].str.strip() != "") & (df["answer"].str.strip() != "")]
            return df.reset_index(drop=True)
        
        ds_raw = getLiveQA_TREC_2017(data_dir)
        ds = reform_LiveQA(ds_raw)
        
    elif dataset_name == "MedicationQA":
        from ...dataLoader.questionAnswering.MedicationQA import getMedicationQA
        def reform_MedicationQA(ds):
            ds = pd.DataFrame(ds)
            ds = ds.rename(columns={"Question": "question", "Answer": "answer"})
            ds = ds.dropna(subset = ["question", "answer"])
            ds = ds.reset_index(drop= True)
            return ds

        ds_raw = getMedicationQA(data_dir)
        ds = reform_MedicationQA(ds_raw)

    elif dataset_name == "MedMCQA":
        from ...dataLoader.questionAnswering.MedMCQA import getMedMCQA
        def reform_MedMCQA(ds):
            ds = ds[0] + ds[1] + ds[2]
            reformed = []
            for entity in ds:
                new_entity = entity
                opa = entity.get("opa")
                opb = entity.get("opb")
                opc = entity.get("opc")
                opd = entity.get("opd")
                cop = entity.get("cop")
                exp = entity.get("exp")
                new_entity["answer"] = f"The choices are: A) {opa}, B) {opb}, C) {opc}, D) {opd}. The correct answer is {cop}, because {exp}"
                reformed.append(new_entity)
            reformed = pd.DataFrame(reformed)
            reformed = reformed.dropna(subset = ["question", "answer"])
            reformed = reformed.reset_index(drop= True)
            return reformed
        
        ds_raw = getMedMCQA(data_dir)
        ds = reform_MedMCQA(ds_raw)

    elif dataset_name == "MedQA-USMLE":
        from ...dataLoader.questionAnswering.MedQA_USMLE import getMedQA_USMLE
        def reform_MedQA_USMLE(ds):
            ds = ds[0] + ds[1] + ds[2]
            reformed = []
            for entity in ds:
                options = entity["options"]
                answer = entity["answer"]
                new_entity = {
                    "question": entity["question"],
                    "options": entity["options"],
                    "old_answer": entity["answer"],
                    "meta_info": entity["meta_info"],
                    "answer_idx": entity["answer_idx"],
                    "answer": f"The options you have are {options}. The correct answer is {answer}."
                }
                reformed.append(new_entity)
            reformed = pd.DataFrame(reformed)
            reformed = reformed.dropna(subset = ["question", "answer"])
            reformed = reformed.reset_index(drop= True)
            return reformed

        ds_raw = getMedQA_USMLE(data_dir, "all")
        ds = reform_MedQA_USMLE(ds_raw)

    elif dataset_name == "PubMedQA":
        from ...dataLoader.questionAnswering.PubMedQA import getPubMedQA
        def reform_PubMedQA(ds):
            ds = pd.DataFrame(ds)
            ds = ds.rename(columns={"QUESTION": "question", "LONG_ANSWER": "answer"})
            ds = ds.dropna(subset = ["question", "answer"])
            ds = ds.reset_index(drop= True)
            return ds
        
        ds_raw = getPubMedQA(data_dir, "all")
        ds = reform_PubMedQA(ds_raw)
    
    return ds


def get_embeddings(texts, model, batch_size = 64):
    """
    Get the embeddings for the given texts. Use batch processing to speed up the process.
    """
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc = "Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            embeddings.extend(model.encode(texts = batch_texts)['text_embeddings'])
    return np.array(embeddings)


def compute_similarity_chunked(embeddings, threshold=0.9, chunk_size=8000):
    """
    Given the embeddings, where each row is the embedding of a data point's text, calculate the similarity between each data point.
    Return the indices of the data points that are similar to each other based on the given threshold.
    Used to deduplicate within a dataset.
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
    Used to deduplicate across datasets.
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


def deduplication_within_dataset_qa(dataset, model, threshold = 0.9):
    """
    Given the dataset, deduplicate the dataset within itself.
    """
    questions = dataset["question"].tolist()
    #answers = dataset["answer"].tolist()

    question_embeddings = get_embeddings(questions, model)
    to_remove_questions = compute_similarity_chunked(question_embeddings, threshold)

    new_dataset = dataset.drop(index = list(to_remove_questions)).reset_index(drop=True)

    answers = new_dataset["answer"].tolist()
    answer_embeddings = get_embeddings(answers, model)
    to_remove_answers = compute_similarity_chunked(answer_embeddings, threshold)

    new_dataset = new_dataset.drop(index = list(to_remove_answers)).reset_index(drop=True)
    return new_dataset, list(to_remove_questions), list(to_remove_answers)


def deduplicate_across_datasets_qa(new_dataset, old_question_embeddings_saved, old_answer_embeddings_saved, model, threshold = 0.9):
    """
    Given the new dataset and the old datasets, deduplicate the new dataset across the old datasets.
    """
   
    old_question_embeddings = []
    old_answer_embeddings = []
    for old_embed in old_question_embeddings_saved:
        old_question_embeddings.extend(old_embed)
    for old_embed in old_answer_embeddings_saved:
        old_answer_embeddings.extend(old_embed)

    # Generate embeddings for new dataset questions and answers
    new_question_embeddings = get_embeddings(new_dataset["question"].tolist(), model)
    new_answer_embeddings = get_embeddings(new_dataset["answer"].tolist(), model)

    # Deduplicate new questions
    to_remove_questions = compute_similarity_between_datasets_chunked(new_question_embeddings, old_question_embeddings, threshold)

    # Deduplicate new answers
    to_remove_answers = compute_similarity_between_datasets_chunked(new_answer_embeddings, old_answer_embeddings, threshold)

    # Combine removal indices
    to_remove = to_remove_questions.union(to_remove_answers)

    # Drop duplicates from new dataset
    deduplicated_new_dataset = new_dataset.drop(index=list(to_remove)).reset_index(drop=True)

    return deduplicated_new_dataset, list(to_remove_questions), list(to_remove_answers)


## Calculate Existing Embeddings
def calculate_and_save_embeddings(dataset, dataset_name, model, save_dir="embeddings_cache", batch_size=128):
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # File paths for embeddings
    question_embedding_file = os.path.join(save_dir, f"{dataset_name}_question_embeddings.pkl")
    answer_embedding_file = os.path.join(save_dir, f"{dataset_name}_answer_embeddings.pkl")

    # Check if embeddings already exist
    if os.path.exists(question_embedding_file) and os.path.exists(answer_embedding_file):
        print(f"Loading cached embeddings for {dataset_name}...")
        with open(question_embedding_file, "rb") as qf:
            question_embeddings = pickle.load(qf)
        with open(answer_embedding_file, "rb") as af:
            answer_embeddings = pickle.load(af)
    else:
        with torch.no_grad():
            # Compute embeddings for questions
            print(f"Generating question embeddings for {dataset_name}...")
            questions = dataset["question"].tolist()
            question_embeddings = []
            for i in tqdm(range(0, len(questions), batch_size), desc="Question Embeddings"):
                batch_questions = questions[i:i + batch_size]
                question_embeddings.extend(model.encode(texts=batch_questions)["text_embeddings"])
            question_embeddings = np.array(question_embeddings)

            # Save question embeddings
            with open(question_embedding_file, "wb") as qf:
                pickle.dump(question_embeddings, qf)
            print(f"Saved question embeddings for {dataset_name}.")

            # Compute embeddings for answers
            print(f"Generating answer embeddings for {dataset_name}...")
            answers = dataset["answer"].tolist()
            answer_embeddings = []
            for i in tqdm(range(0, len(answers), batch_size), desc="Answer Embeddings"):
                batch_answers = answers[i:i + batch_size]
                answer_embeddings.extend(model.encode(texts=batch_answers)["text_embeddings"])
            answer_embeddings = np.array(answer_embeddings)

            # Save answer embeddings
            with open(answer_embedding_file, "wb") as af:
                pickle.dump(answer_embeddings, af)
            print(f"Saved answer embeddings for {dataset_name}.")

    return {"questions": question_embeddings, "answers": answer_embeddings}
