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
def getCORD19(path, subtitle):
    try:
        data_path = os.path.join(path, "CORD19")
        if os.path.exists(data_path):
            ds = datasets.load_from_disk(data_path)
        else:
            ds = datasets.load_dataset("allenai/cord19", subtitle, trust_remote_code=True)
            ds.save_to_disk(data_path)
            
        df_train = ds["train"].to_pandas()

        dataset_train = df_train.to_dict(orient='records')

        return dataset_train

    except Exception as e:
        breakpoint()
        print_sys(f"error: {e}")