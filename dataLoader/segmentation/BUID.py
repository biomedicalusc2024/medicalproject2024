import os
import warnings

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

BUID_SUBTITLE = ["benign", "malignant", "normal"]


# tested by tjl 2025/1/22
def getBUID(path, subtitle):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/aryashah2k/breast-ultrasound-images-dataset"]
    if subtitle not in BUID_SUBTITLE:
        raise AttributeError(f"Please enter dataset name in BUID-subset format and select the subsection of BUID in {BUID_SUBTITLE}")
    return datasetLoad(urls=urls, subtitle=subtitle, path=path, datasetName="BUID")


def datasetLoad(urls, subtitle, path, datasetName):
    datasetPath = os.path.join(path, datasetName)
    zip_path = os.path.join(datasetPath, "raw.zip")
    try:
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath, subtitle)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], zip_path, datasetPath)
            return loadLocalFiles(datasetPath, subtitle)

    except Exception as e:
        print_sys(f"Error: {e}")


def loadLocalFiles(path, subtitle):
    basePath = os.path.join(path, "Dataset_BUSI_with_GT", subtitle)
    
    dataset = []
    all_mask = [fn for fn in os.listdir(basePath) if "mask" in fn]
    for mask in all_mask:
        mask_path = os.path.join(basePath, mask)
        source = "".join(mask.split("_mask"))
        source_path = os.path.join(basePath, source)
        dataset.append({"source_path":source_path, "mask_path":mask_path})

    return dataset
