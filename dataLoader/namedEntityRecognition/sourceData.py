import os
import re
import requests
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

warnings.filterwarnings("ignore")

from ..utils import print_sys


def getSourceData(path):
    cache_dir = os.path.join(path, "SourceData")
    ds = load_dataset("EMBO/SourceData", "NER", cache_dir=cache_dir, version="2.0.3", trust_remote_code=True)
    df_train = ds["train"].to_pandas()
    df_test = ds["test"].to_pandas()
    df_val = ds["validation"].to_pandas()
    source_cols = ['words', 'labels', 'text']
    target_cols = ['tag_mask']

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
