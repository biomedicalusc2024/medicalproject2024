import os
from datasets import load_dataset

# Error when loading datasets, not clear how to fix

def getPubMed(path):
    cache_dir = os.path.join(path, "PubMed")
    pubmed = load_dataset("ncbi/pubmed", cache_dir=cache_dir, trust_remote_code=True, split="train[10:100]")
    breakpoint()