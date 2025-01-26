import os
import warnings

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/23
def getKvasir(path):
    urls = {
        "Kvasir_v2": "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip",
        "Kvasir_v2_feature": "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2-features.zip",
    }

    return datasetLoad(urls=urls, path=path, datasetName="Kvasir")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            for _, url in urls.items():
                file_name = url.split("/")[-1]
                file_path = os.path.join(datasetPath, file_name)
                download_file(url, file_path, datasetPath)
            return loadLocalFiles(datasetPath)
    
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    dataset = []
    data_base_path = os.path.join(path, "kvasir-dataset-v2")
    feature_base_path = os.path.join(path, "kvasir-dataset-v2-features")

    for sub in os.listdir(data_base_path):
        data_subpath = os.path.join(data_base_path, sub)
        feature_subpath = os.path.join(feature_base_path, sub)
        for fn in os.listdir(data_subpath):
            fid = fn.split(".")[0]
            data_fn = os.path.join(data_subpath, f"{fid}.jpg")
            feature_fn = os.path.join(feature_subpath, f"{fid}.features")
            if os.path.exists(data_fn) and os.path.exists(feature_fn):
                dataset.append({"data_path":data_fn,"feature_path":feature_fn})

    return dataset
