import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# need to submit files to acquire permission, skipped
def getBKAI_IGH(path):
    urls = "bkai-igh-neopolyp"
    return datasetLoad(urls=urls, path=path, datasetName="BKAI_IGH")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        zipPath = os.path.join(datasetPath,'raw.zip')
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], zipPath, datasetPath)
            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    breakpoint()