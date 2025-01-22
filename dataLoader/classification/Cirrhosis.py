import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/21
def getCirrhosis(path):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/fedesoriano/cirrhosis-prediction-dataset"]
    return datasetLoad(urls=urls, path=path, datasetName="Cirrhosis")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            download_file(urls[0], datasetZip, datasetPath)
            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df = pd.read_csv(os.path.join(path, "cirrhosis.csv"))
    dataset = df.to_dict(orient='records')
    return dataset