import os
import shutil
import zipfile
import requests
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/19
def getMedMCQA(path):
    urls = ["https://drive.usercontent.google.com/download?id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky&export=download&authuser=0&confirm=t&uuid=577323dd-78af-4a4e-bb4d-e071a819bcdc&at=APvzH3qNqiLKFaPCPIXbR5WsTeqI:1736307787694"]
    return datasetLoad(urls=urls, path=path, datasetName="MedMCQA")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(path, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            download_file(urls[0], datasetFile, datasetPath)
            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df_train = pd.read_json(os.path.join(path, "train.json"), lines=True)
    df_test = pd.read_json(os.path.join(path, "test.json"), lines=True)
    df_val = pd.read_json(os.path.join(path, "dev.json"), lines=True)

    trainset = df_train.to_dict(orient='records')
    testset = df_test.to_dict(orient='records')
    valset = df_val.to_dict(orient='records')

    return trainset, testset, valset