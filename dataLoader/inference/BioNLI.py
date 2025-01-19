import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/18
def getBioNLI(path):
    urls = [
        "https://drive.usercontent.google.com/u/0/uc?id=1pqXyU4E13-foNHdH8uBQ_YnY7i03nG1J&export=download",
        "https://drive.usercontent.google.com/u/0/uc?id=1VYM7swTcrYHv8nUMSKmcm3IBgPNROG0w&export=download",
        "https://drive.usercontent.google.com/u/0/uc?id=1qpTqcmSHoF3P89Vf57agEBdyOGQJQpy5&export=download",
    ]
    
    return datasetLoad(urls=urls, path=path, datasetName="BioNLI")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            
            download_file(urls[0], os.path.join(datasetPath, "train.csv"))
            download_file(urls[1], os.path.join(datasetPath, "test.csv"))
            download_file(urls[2], os.path.join(datasetPath, "dev.csv"))

            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df_train = pd.read_csv(os.path.join(path, "train.csv"))
    df_test = pd.read_csv(os.path.join(path, "test.csv"))
    df_val = pd.read_csv(os.path.join(path, "dev.csv"))

    dataset_train = df_train.to_dict(orient='records')
    dataset_test = df_test.to_dict(orient='records')
    dataset_val = df_val.to_dict(orient='records')

    return dataset_train, dataset_test, dataset_val
    