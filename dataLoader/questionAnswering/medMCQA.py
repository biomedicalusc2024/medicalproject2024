import os
import shutil
import zipfile
import requests
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getMedMCQA(path):
    url = "https://drive.usercontent.google.com/download?id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky&export=download&authuser=0&confirm=t&uuid=577323dd-78af-4a4e-bb4d-e071a819bcdc&at=APvzH3qNqiLKFaPCPIXbR5WsTeqI:1736307787694"
    return datasetLoad(url=url, path=path, datasetName="MedMCQA")


def datasetLoad(url, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(path, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            print_sys("Downloading...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                os.makedirs(datasetPath, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(datasetFile, "wb") as file:
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

                with zipfile.ZipFile(datasetFile, "r") as z:
                    z.extractall(datasetPath)
                print_sys("Extraction complete.")
                os.remove(datasetFile)
                shutil.rmtree(os.path.join(datasetPath, "__MACOSX"))

                return loadLocalFiles(datasetPath)
            else:
                print_sys("Connection error, please check the internet.")
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df_train = pd.read_json(os.path.join(path, "train.json"), lines=True)
    df_test = pd.read_json(os.path.join(path, "test.json"), lines=True)
    df_val = pd.read_json(os.path.join(path, "dev.json"), lines=True)

    source_cols1 = ['question', 'exp', 'opa', 'opb', 'opc', 'opd', 'subject_name', 'topic_name', 'id', 'choice_type']
    source_cols2 = ['question', 'opa', 'opb', 'opc', 'opd', 'subject_name', 'topic_name', 'id', 'choice_type']
    target_cols = ['cop']

    trainset = {
        "source": df_train[source_cols1].to_numpy().tolist(),
        "target": df_train[target_cols].to_numpy().tolist(),
    }
    testset = {
        "source": df_test[source_cols2].to_numpy().tolist(),
        "target": [[]] * len(df_test),
    }
    valset = {
        "source": df_val[source_cols1].to_numpy().tolist(),
        "target": df_val[target_cols].to_numpy().tolist(),
    }

    return trainset, testset, valset