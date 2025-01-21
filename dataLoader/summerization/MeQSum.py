import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/20
def getMeQSum(path):
    urls = ["https://huggingface.co/datasets/sumedh/MeQSum/resolve/main/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx"]
    return datasetLoad(urls=urls, path=path, datasetName="MeQSum")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(datasetPath, "MeQSum.xlsx")
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
    df = pd.read_excel(path)
    dataset = df.to_dict(orient='records')
    return dataset