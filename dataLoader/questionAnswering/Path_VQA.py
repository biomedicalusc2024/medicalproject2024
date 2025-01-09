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


def getPath_VQA(path):
    try:
        data_path = os.path.join(path, "Path_VQA")
        if os.path.exists(data_path):
            ds = datasets.load_from_disk(data_path)
        else:
            ds = datasets.load_dataset("flaviagiammarino/path-vqa", trust_remote_code=True)
            ds.save_to_disk(data_path)
            
        df_train = ds["train"].to_pandas()
        df_test = ds["test"].to_pandas()
        df_val = ds["validation"].to_pandas()
        source_cols = ['image', 'question']
        target_cols = ['answer']

        dataset_train = {
            "source": df_train[source_cols].to_numpy().tolist(),
            "target": df_train[target_cols].to_numpy().tolist()
        }

        dataset_test = {
            "source": df_test[source_cols].to_numpy().tolist(),
            "target": df_test[target_cols].to_numpy().tolist()
        }

        dataset_val = {
            "source": df_val[source_cols].to_numpy().tolist(),
            "target": df_val[target_cols].to_numpy().tolist()
        }
        return dataset_train, dataset_test, dataset_val

    except Exception as e:
        print_sys(f"error: {e}")