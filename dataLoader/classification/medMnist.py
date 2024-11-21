import os
import requests
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys

ALLOPTIONS = [
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

def getMedMnist(path, sub=""):
    if sub in ALLOPTIONS:
        url = f"https://zenodo.org/records/10519652/files/{sub}.npz?download=1"
    else:
        raise AttributeError(f"Please enter dataset name in MedMnist-subset format and select the subsection of MedMnist in {ALLOPTIONS}")
    return datasetLoad(url=url, path=path, sub=sub, datasetName="MedMnist")

def datasetLoad(url, path, sub, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(datasetPath, f"{sub}.npz")
        if os.path.exists(datasetFile):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetFile)
        else:
            print_sys("Downloading...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                os.makedirs(datasetPath, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(datasetFile, "wb") as file:
                    if total_size == 0:
                        pbar = None
                    else:
                        pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))
                    if pbar:
                        pbar.close()
                print_sys("Download complete.")
                return loadLocalFiles(datasetFile)
            else:
                print_sys("Connection error, please check the internet.")
    except Exception as e:
        print_sys(f"error: {e}")

def loadLocalFiles(path):
    data = np.load(path)
    trainset = {
        "source": data["train_images"],
        "target": data["train_labels"]
    }
    testset = {
        "source": data["test_images"],
        "target": data["test_labels"]
    }
    valset = {
        "source": data["val_images"],
        "target": data["val_labels"]
    }
    return trainset, testset, valset