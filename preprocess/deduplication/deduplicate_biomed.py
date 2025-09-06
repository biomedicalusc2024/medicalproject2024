import os
import pickle
import pandas as pd
from subprocess import check_output

from .biomed_dedup_util import (
    load_dataset, 
    deduplicate_within_dataset,
    deduplicate_between_datasets,
    calculate_and_save_embeddings
)

# all the available datasets, can be changed or updated. But the one here are tested for preprocessingand working.
AVAILABLE_DATASETS = [
    "bc5cdr",
    "BioNLI",
    "CORD19",
    # "DDCICorpus",
    "hoc",
    # "pubmed", 
    "SourceData",
    # "trec_covid",
]

# all the available models, maybe can expand later.
AVAILABLE_MODELS = [
    "MedImageInsight"
]

def load_model(model_name):
    """Load the embedding model"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(AVAILABLE_MODELS)}")
    
    # maybe add other models here later. Or we provide download links for the models.
    if model_name == "MedImageInsight":
        dir_path = os.path.dirname(__file__)
        if "MedImageInsights" not in os.listdir(dir_path):
            check_output(["git", "clone", "https://huggingface.co/lion-ai/MedImageInsights", f"{dir_path}/MedImageInsights"])

        import sys
        sys.path.append(f"{dir_path}/MedImageInsights")
        from .MedImageInsights.medimageinsightmodel import MedImageInsight

        model = MedImageInsight(
            model_dir=f"{os.path.dirname(__file__)}/MedImageInsights/2024.09.27",
            # model_dir="MedImageInsights/2024.09.27",
            vision_model_name="medimageinsigt-v1.0.0.pt",
            language_model_name="language_model.pth"
        )
        model.load_model()
        return model

    

def process_dataset(dataset_name, data_dir, save_dir, embeddings_dir, model, threshold, test):
    """Process a single dataset through deduplication pipeline"""
    
    try:
        # Load dataset
        ds = load_dataset(dataset_name, data_dir)
        if test:
            ds = ds.sample(min(320, len(ds)), random_state=42).reset_index(drop=True)

        col_df = pd.read_csv(f"{os.path.dirname(__file__)}/col.csv")
        col_list = col_df.loc[col_df["dataset_name"] == dataset_name, "column_name"].tolist()[0].split(', ')
        
        # Within dataset deduplication
        deduplicated_data, _ = deduplicate_within_dataset(ds, col_list, model, threshold)
        
        # load old_embeddings
        old_embeddings = []
        for file in os.listdir(embeddings_dir):
            if file.endswith(".pkl"):
                old_embeddings.append(pickle.load(open(os.path.join(embeddings_dir, file), "rb")))
        
        # Between dataset deduplication
        deduplicated_data, _ = deduplicate_between_datasets(deduplicated_data, col_list, model, old_embeddings, threshold)
        
        # Save deduplicated data
        save_path = os.path.join(save_dir, f"{dataset_name}_deduplicated.csv")
        deduplicated_data.to_csv(save_path, index=False)

        # save embeddings
        calculate_and_save_embeddings(deduplicated_data, dataset_name, model, col_list, embeddings_dir)
        return deduplicated_data
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        return None
    
    

def deduplicate_biomed(datasets, data_dir, save_dir, test=False, model_name="MedImageInsight", threshold=0.9):
    '''
    datasets(List(Str)): which datasets to load
    '''
    # Download/verify model
    model = load_model(model_name)
    
    # Create save directory
    qa_save_dir = os.path.join(save_dir, "deduplicate_biomed")
    if not os.path.exists(qa_save_dir):
        os.makedirs(qa_save_dir, exist_ok=True)
    embeddings_dir = os.path.join(qa_save_dir, "embedding_biomed")
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir, exist_ok=True)
    
    # Determine datasets to process
    if "all" in datasets:
        datasets_to_process = AVAILABLE_DATASETS
    else:
        for ds in datasets:
            if ds not in AVAILABLE_DATASETS:
                raise AttributeError(f"{ds} not supported for biomed deduplication, please select in {AVAILABLE_DATASETS}")
        datasets_to_process = datasets
    
    deduplicate_results = {}
    for dataset in datasets_to_process:
        try:
            result = process_dataset(
                dataset,
                data_dir,
                qa_save_dir,
                embeddings_dir,
                model,
                threshold,
                test
            )
            if result is None:
                print(f"Failed to process {dataset}")
            else:
                print(f"Successfully processed {dataset}")
                deduplicate_results[dataset] = result
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue
    
    return deduplicate_results
    
