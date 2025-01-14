import os
import json
import shutil
import rarfile
import tarfile
import zipfile
import requests
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys


def getMOSES(path):
    urls = [
        "https://media.githubusercontent.com/media/molecularsets/moses/refs/heads/master/data/train.csv",
        "https://media.githubusercontent.com/media/molecularsets/moses/refs/heads/master/data/test.csv",
    ]
    return datasetLoad(urls, path=path, datasetName="MOSES")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath)
            for url in urls:
                fn = url.split("/")[-1]
                download_file(url, os.path.join(datasetPath, fn))

            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df_train = pd.read_csv(os.path.join(path, "train.csv"))
    df_test = pd.read_csv(os.path.join(path, "test.csv"))

    source_cols = ["SMILES"]
    target_cols = []

    trainset = {
        "source": df_train[source_cols].to_numpy().tolist(),
        "target": df_train[target_cols].to_numpy().tolist(),
    }
    testset = {
        "source": df_test[source_cols].to_numpy().tolist(),
        "target": df_test[target_cols].to_numpy().tolist(),
    }
    
    return trainset, testset
    


def download_file(url, destination, extractionPath=None):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(destination, "wb") as file:
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
            print_sys(f"Download {destination} complete.")

            if extractionPath:
                if "zip" in destination:
                    with zipfile.ZipFile(destination, "r") as z:
                        z.extractall(extractionPath)
                    print_sys("Extraction complete.")
                    os.remove(destination)
                elif "rar" in destination:
                    with rarfile.RarFile(destination) as rf:
                        rf.extractall(extractionPath)
                    print_sys("Extraction complete.")
                    os.remove(destination)
                elif "tar" in destination:
                    if "gz" in destination:
                        with tarfile.open(destination, 'r:gz') as tar:
                            tar.extractall(extractionPath)
                        print_sys("Extraction complete.")
                        os.remove(destination)
                    else:
                        with tarfile.open(destination, 'r') as tar:
                            tar.extractall(extractionPath)
                        print_sys("Extraction complete.")
                        os.remove(destination)

    except Exception as e:
        print_sys(f"error: {e}")