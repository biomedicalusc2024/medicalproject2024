import os
import requests
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getBioNLI(path):
    url = ["https://drive.usercontent.google.com/u/0/uc?id=1pqXyU4E13-foNHdH8uBQ_YnY7i03nG1J&export=download",
           "https://drive.usercontent.google.com/u/0/uc?id=1VYM7swTcrYHv8nUMSKmcm3IBgPNROG0w&export=download",
           "https://drive.usercontent.google.com/u/0/uc?id=1qpTqcmSHoF3P89Vf57agEBdyOGQJQpy5&export=download"]
    
    return datasetLoad(url=url, path=path, datasetName="BioNLI")


def datasetLoad(url, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            print_sys("Downloading...")
            response = requests.get(url[0], stream=True)
            if response.status_code == 200:
                os.makedirs(datasetPath, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(os.path.join(datasetPath, "train.csv"), "wb") as file:
                    if total_size == 0:
                        pbar = None
                    else:
                        pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))
                    if pbar:
                        pbar.close()
                print_sys("Download complete.")
            else:
                print_sys("Connection error, please check the internet.")

            response = requests.get(url[1], stream=True)
            if response.status_code == 200:
                os.makedirs(datasetPath, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(os.path.join(datasetPath, "test.csv"), "wb") as file:
                    if total_size == 0:
                        pbar = None
                    else:
                        pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))
                    if pbar:
                        pbar.close()
                print_sys("Download complete.")
            else:
                print_sys("Connection error, please check the internet.")

            response = requests.get(url[2], stream=True)
            if response.status_code == 200:
                os.makedirs(datasetPath, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(os.path.join(datasetPath, "dev.csv"), "wb") as file:
                    if total_size == 0:
                        pbar = None
                    else:
                        pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))
                    if pbar:
                        pbar.close()
                print_sys("Download complete.")
            else:
                print_sys("Connection error, please check the internet.")

            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    source_cols = ['supp_set', 'conclusion', 'ori_conclusion']
    target_cols = ['label_cat']

    df_train = pd.read_csv(os.path.join(path, "train.csv"))
    df_test = pd.read_csv(os.path.join(path, "test.csv"))
    df_val = pd.read_csv(os.path.join(path, "dev.csv"))

    dataset_train = {
        "source": df_train[source_cols].to_numpy().tolist(),
        "target": df_train[target_cols].to_numpy().tolist(),
    }
    dataset_test = {
        "source": df_test[source_cols].to_numpy().tolist(),
        "target": df_test[target_cols].to_numpy().tolist(),
    }
    dataset_val = {
        "source": df_val[source_cols].to_numpy().tolist(),
        "target": df_val[target_cols].to_numpy().tolist(),
    }

    return dataset_train, dataset_test, dataset_val
    