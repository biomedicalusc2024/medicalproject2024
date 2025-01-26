import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

MedMnist_SUBTITLE = [
    "adrenalmnist3d", "adrenalmnist3d_64", 
    "bloodmnist", "bloodmnist_128", "bloodmnist_224", "bloodmnist_64",
    "breastmnist", "breastmnist_128", "breastmnist_224", "breastmnist_64",
    "chestmnist", "chestmnist_128", "chestmnist_224", "chestmnist_64",
    "dermamnist", "dermamnist_128", "dermamnist_224", "dermamnist_64",
    "fracturemnist3d", "fracturemnist3d_64", 
    "nodulemnist3d", "nodulemnist3d_64",
    "octmnist", "octmnist_128", "octmnist_224", "octmnist_64", 
    "organamnist", "organamnist_128", "organamnist_224", "organamnist_64",
    "organcmnist", "organcmnist_128", "organcmnist_224", "organcmnist_64",
    "organmnist3d", "organmnist3d_64", 
    "organsmnist", "organsmnist_128", "organsmnist_224", "organsmnist_64",
    "pathmnist", "pathmnist_128", "pathmnist_224", "pathmnist_64", 
    "pneumoniamnist", "pneumoniamnist_128", "pneumoniamnist_224", "pneumoniamnist_64",
    "retinamnist", "retinamnist_128", "retinamnist_224", "retinamnist_64",
    "synapsemnist3d", "synapsemnist3d_64",
    "tissuemnist", "tissuemnist_128", "tissuemnist_224", "tissuemnist_64", 
    "vesselmnist3d", "vesselmnist3d_64"
]


# tested by tjl 2025/1/21
def getMedMnist(path, sub=""):
    if sub in MedMnist_SUBTITLE:
        urls = [f"https://zenodo.org/records/10519652/files/{sub}.npz?download=1"]
    else:
        raise AttributeError(f"Please enter dataset name in MedMnist-subset format and select the subsection of MedMnist in {MedMnist_SUBTITLE}")
    return datasetLoad(urls=urls, path=path, sub=sub, datasetName="MedMnist")


def datasetLoad(urls, path, sub, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(datasetPath, f"{sub}.npz")
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
    data = np.load(path)
    trainset = [{"image":img, "label":label} for img,label in zip(data["train_images"],data["train_labels"])]
    testset = [{"image":img, "label":label} for img,label in zip(data["test_images"],data["test_labels"])]
    valset = [{"image":img, "label":label} for img,label in zip(data["val_images"],data["val_labels"])]
    return trainset, testset, valset