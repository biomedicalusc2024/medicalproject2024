import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/19
def getVQA_RAD(path):
    urls = ["https://files.osf.io/v1/resources/89kps/providers/osfstorage/?zip="]
    return datasetLoad(urls=urls, path=path, datasetName="VQA_RAD")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "raw.zip")
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
    img_path = os.path.join(path, "VQA_RAD Image Folder")
    df = pd.read_json(os.path.join(path, "VQA_RAD Dataset Public.json"))
    df["image_name"] = df["image_name"].apply(lambda x: os.path.join(img_path,x))
    dataset = df.to_dict(orient='records')
    return dataset