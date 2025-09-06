import os
import warnings

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/24
def getLiTS(path):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/harshwardhanbhangale/lits-dataset"]
    return datasetLoad(urls=urls, path=path, datasetName="LiTS")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        zipPath = os.path.join(datasetPath,'raw.zip')
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            try:
                download_file(urls[0], zipPath, datasetPath)
            except Exception as e:
                print_sys(f"error: {e}")
            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    train_image_path = os.path.join(path, "train_images", "train_images")
    train_mask_path = os.path.join(path, "train_masks", "train_masks")
    dataset = []

    for fn in os.listdir(train_image_path):
        image_fn = os.path.join(train_image_path, fn)
        mask_fn = os.path.join(train_mask_path, fn)
        if os.path.exists(image_fn) and os.path.exists(mask_fn):
            dataset.append({"image_path":image_fn,"mask_path":mask_fn})
    
    return dataset