import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/18
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

    trainset = df_train.to_dict(orient='records')
    testset = df_test.to_dict(orient='records')
    
    return trainset, testset
    