import os
import requests
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getROND(path):
    url = "https://raw.githubusercontent.com/Mayo-Clinic-RadOnc-Foundation-Models/Radiation-Oncology-NLP-Database/main/1-Logic%20Reasoning/Logic_Reasoning.csv"
    return datasetLoad(url=url, path=path, datasetName="ROND")


def datasetLoad(url, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(datasetPath, "ROND_LogicReasoning.csv")
        if os.path.exists(datasetFile):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetFile)
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
                return loadLocalFiles(datasetFile)
            else:
                print_sys("Connection error, please check the internet.")
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df = pd.read_csv(path, header=None, names=['source', 'target'])
    return {
        "source": df["source"].to_list(),
        "target": df["target"].to_list(),
    }