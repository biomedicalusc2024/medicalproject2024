import os
import shutil
import requests
import warnings
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getPMC_VQA(path):
    url_data = "https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/images.zip?download=true"
    url_train = "https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/train.csv?download=true"
    url_test = "https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/test.csv?download=true"
    return datasetLoad(url_data, url_train, url_test, path=path, datasetName="PMC-VQA")


def datasetLoad(url_data, url_train, url_test, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath, url_train, url_test)
        else:
            print_sys("Downloading...")
            response = requests.get(url_data, stream=True)
            if response.status_code == 200:
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

                return loadLocalFiles(datasetPath, url_train, url_test)
            else:
                print_sys("Connection error, please check the internet.")
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path, url_train, url_test):
    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")
    if not os.path.exists(train_path):
        try:
            response = requests.get(url_train, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))

                with open(train_path, "wb") as file:
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
                print_sys("Download train split complete.")
        except Exception as e:
            print_sys(f"error: {e}")

    if not os.path.exists(test_path):
        try:
            response = requests.get(url_test, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))

                with open(test_path, "wb") as file:
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
                print_sys("Download test split complete.")
        except Exception as e:
            print_sys(f"error: {e}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    source_cols = ['Figure_path', 'Question', 'Choice A', 'Choice B', 'Choice C', 'Choice D']
    target_cols = ['Answer', 'Answer_label']
    
    trainset = {
        "source": df_train[source_cols].to_numpy().tolist(),
        "target": df_train[target_cols].to_numpy().tolist()
    }
    testset = {
        "source": df_test[source_cols].to_numpy().tolist(),
        "target": df_test[target_cols].to_numpy().tolist()
    }
    return trainset, testset