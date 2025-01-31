import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


def getCheXmask(path):
    urls = ["https://physionet.org/static/published-projects/chexmask-cxr-segmentation-data/chexmask-database-a-large-scale-dataset-of-anatomical-segmentation-masks-for-chest-x-ray-images-0.1.zip"]
    return datasetLoad(urls=urls, path=path, datasetName="CheXmask")


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
    basePath = os.path.join(path, "chexmask-database-a-large-scale-dataset-of-anatomical-segmentation-masks-for-chest-x-ray-images-0.1")
    originPath = os.path.join(basePath, "OriginalResolution")
    preprocessedPath = os.path.join(basePath, "Preprocessed")
    
    df = pd.read_csv(os.path.join(originPath, "CheXpert.csv"))
    breakpoint()