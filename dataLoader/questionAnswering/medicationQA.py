import os
import re
import requests
import warnings
import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys


def getMedicationQA(path):
    try:
        data_path = os.path.join(path, "MedicationQA")
        if os.path.exists(data_path):
            ds = datasets.load_from_disk(data_path)
        else:
            ds = datasets.load_dataset("truehealth/medicationqa", trust_remote_code=True)
            ds.save_to_disk(data_path)
            
        df = ds["train"].to_pandas()
        source_cols = ['Question', 'Focus (Drug)', 'Question Type', 'Section Title', 'URL']
        target_cols = ['Answer']

        dataset = {
            "source": df[source_cols].to_numpy().tolist(),
            "target": df[target_cols].to_numpy().tolist()
        }

        return dataset

    except Exception as e:
        print_sys(f"error: {e}")