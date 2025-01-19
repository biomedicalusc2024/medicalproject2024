import os
import requests
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/18
def getROND(path):
    urls = ["https://raw.githubusercontent.com/Mayo-Clinic-RadOnc-Foundation-Models/Radiation-Oncology-NLP-Database/main/1-Logic%20Reasoning/Logic_Reasoning.csv"]
    return datasetLoad(urls=urls, path=path, datasetName="ROND")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(datasetPath, "ROND_LogicReasoning.csv")
        if os.path.exists(datasetFile):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetFile)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], datasetFile)
            return loadLocalFiles(datasetFile)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df = pd.read_csv(path)
    df["Question"] = df["Question"].apply(lambda x: x.split(".")[1].strip())
    dataset = df.to_dict(orient='records')
    return dataset