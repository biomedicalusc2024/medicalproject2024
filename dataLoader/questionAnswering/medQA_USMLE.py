import os
import shutil
import requests
import zipfile
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getMedQA_USMLE(path, subtitle):
    url = "https://www.kaggle.com/api/v1/datasets/download/moaaztameer/medqa-usmle"
    return datasetLoad(url=url, subtitle=subtitle, path=path, datasetName="MedQA_USMLE")


def datasetLoad(url, subtitle, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(datasetPath, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath, subtitle)
        else:
            print_sys("Downloading...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                os.makedirs(datasetPath, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(datasetZip, "wb") as file:
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

                with zipfile.ZipFile(datasetZip, "r") as z:
                    z.extractall(datasetPath)
                print_sys("Extraction complete.")
                os.remove(datasetZip)

                return loadLocalFiles(datasetPath, subtitle)
            else:
                print_sys("Connection error, please check the internet.")
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path, subtitle):
    basePath = os.path.join(path, "MedQA-USMLE", "questions")
    if subtitle == "mainland":
        basePath = os.path.join(basePath, "Mainland")
    elif subtitle == "taiwan":
        basePath = os.path.join(basePath, "Taiwan")
    elif subtitle == "us":
        basePath = os.path.join(basePath, "US")
    
    df_train = pd.read_json(os.path.join(basePath, "train.jsonl"), lines=True)
    df_test = pd.read_json(os.path.join(basePath, "test.jsonl"), lines=True)
    df_val = pd.read_json(os.path.join(basePath, "dev.jsonl"), lines=True)

    source_cols = ['question', 'options', 'meta_info']
    target_cols = ['answer', 'answer_idx']

    trainset = {
        "source": df_train[source_cols].to_numpy().tolist(),
        "target": df_train[target_cols].to_numpy().tolist(),
    }
    testset = {
        "source": df_test[source_cols].to_numpy().tolist(),
        "target": df_test[target_cols].to_numpy().tolist(),
    }
    valset = {
        "source": df_val[source_cols].to_numpy().tolist(),
        "target": df_val[target_cols].to_numpy().tolist(),
    }

    return trainset, testset, valset