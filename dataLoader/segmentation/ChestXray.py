import os
import warnings

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/24
def getChestXray(path):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/nikhilpandey360/chest-xray-masks-and-labels"]
    return datasetLoad(urls=urls, path=path, datasetName="ChestXray")


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
    basePath = os.path.join(path, "Lung Segmentation")
    imagePath = os.path.join(basePath, "CXR_png")
    maskPath = os.path.join(basePath, "masks")
    clinicalPath = os.path.join(basePath, "ClinicalReadings")

    images = os.listdir(imagePath)
    masks = os.listdir(maskPath)
    clinicals = os.listdir(clinicalPath)

    dataset = []
    for fn in images:
        image_fn = os.path.join(imagePath, fn)
        ele = {"image_path": image_fn}
        fid = fn.split(".")[0]

        mask_fn = [mask for mask in masks if fid in mask]
        if len(mask_fn) > 0:
            ele["mask_path"] = os.path.join(maskPath, mask_fn[0])
        
        clinical_fn = [clinical for clinical in clinicals if fid in clinical]
        if len(clinical_fn) > 0:
            ele["clinical_path"] = os.path.join(clinicalPath, clinical_fn[0])

        dataset.append(ele)
        
    return dataset