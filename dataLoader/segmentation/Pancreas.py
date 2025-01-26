import os
import warnings

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/23
def getPancreas(path):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/tahsin/pancreasct-dataset"]
    return datasetLoad(urls=urls, path=path, datasetName="PancreasCT")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        zipPath = os.path.join(datasetPath, "raw.zip")
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
    dataset = []
    imagePath = os.path.join(path, "images")
    labelPath = os.path.join(path, "labels")

    for fn in os.listdir(labelPath):
        image_fn = os.path.join(imagePath, fn)
        label_fn = os.path.join(labelPath, fn)
        if os.path.exists(image_fn) and os.path.exists(label_fn):
            dataset.append({"image_path":image_fn,"label_path":label_fn})
    
    return dataset