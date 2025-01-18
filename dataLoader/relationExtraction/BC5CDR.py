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


# tested by tjl 2025/1/17
def getBC5CDR(path):
    try:
        data_path = os.path.join(path, "BC5CDR")
        if os.path.exists(data_path):
            ds = datasets.load_from_disk(data_path)
        else:
            ds = datasets.load_dataset("bigbio/bc5cdr", trust_remote_code=True)
            ds.save_to_disk(data_path)
            
        df_train = ds["train"].to_pandas()
        df_test = ds["test"].to_pandas()
        df_val = ds["validation"].to_pandas()

        dataset_train = df_train.to_dict(orient='records')
        dataset_test = df_test.to_dict(orient='records')
        dataset_val = df_val.to_dict(orient='records')

        return dataset_train, dataset_test, dataset_val

    except Exception as e:
        print_sys(f"error: {e}")