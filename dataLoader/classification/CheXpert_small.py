import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/21
def getCheXpert_small(path):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/ashery/chexpert"]
    return datasetLoad(urls=urls, path=path, datasetName="CheXpert_small")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(datasetPath, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], datasetZip, datasetPath)
            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    def check(x):
        return os.path.exists(x)
    df_train = pd.read_csv(os.path.join(path,"train.csv"))
    df_valid = pd.read_csv(os.path.join(path,"valid.csv"))
    df_train["Path"] = df_train["Path"].apply(lambda x: os.path.join(path,"/".join(x.split("/")[1:])))
    df_valid["Path"] = df_valid["Path"].apply(lambda x: os.path.join(path,"/".join(x.split("/")[1:])))
    df_train = df_train[df_train["Path"].apply(check)]
    df_valid = df_valid[df_valid["Path"].apply(check)]
    trainset = df_train.to_dict(orient='records')
    validset = df_valid.to_dict(orient='records')
    return trainset, validset