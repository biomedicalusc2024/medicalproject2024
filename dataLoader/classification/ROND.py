import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/21
def getROND(path):
    urls = ["https://raw.githubusercontent.com/Mayo-Clinic-RadOnc-Foundation-Models/Radiation-Oncology-NLP-Database/main/2-Text%20Classification/Text_Classification.csv"]
    return datasetLoad(urls=urls, path=path, datasetName="ROND")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(datasetPath, "ROND_Classification.csv")
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
    df = pd.read_csv(path, header=None, names=["data", "label"])
    dataset = df.to_dict(orient="records")
    return dataset