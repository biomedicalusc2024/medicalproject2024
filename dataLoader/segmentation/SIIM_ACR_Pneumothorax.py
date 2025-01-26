import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/25
def getSIIM_ACR(path):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/vbookshelf/pneumothorax-chest-xray-images-and-masks"]
    return datasetLoad(urls=urls, path=path, datasetName="SIIM_ACR")


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
    basePath = os.path.join(path, "siim-acr-pneumothorax")
    train_df = pd.read_csv(os.path.join(basePath, "stage_1_train_images.csv"))
    test_df = pd.read_csv(os.path.join(basePath, "stage_1_test_images.csv"))
    
    image_path = os.path.join(basePath, "png_images")
    mask_path = os.path.join(basePath, "png_masks")

    images = os.listdir(image_path)
    masks = os.listdir(mask_path)

    train_df = train_df[train_df["new_filename"].isin(images)]
    test_df = test_df[test_df["new_filename"].isin(images)]
    train_df = train_df[train_df["new_filename"].isin(masks)]
    test_df = test_df[test_df["new_filename"].isin(masks)]

    train_df.rename(columns={'new_filename': 'image_path'}, inplace=True)
    test_df.rename(columns={'new_filename': 'image_path'}, inplace=True)

    train_df["mask_path"] = train_df["image_path"].apply(lambda x: os.path.join(mask_path,x))
    train_df["image_path"] = train_df["image_path"].apply(lambda x: os.path.join(image_path,x))
    test_df["mask_path"] = test_df["image_path"].apply(lambda x: os.path.join(mask_path,x))
    test_df["image_path"] = test_df["image_path"].apply(lambda x: os.path.join(image_path,x))

    trainset = train_df.to_dict(orient='records')
    testset = test_df.to_dict(orient='records')
    return trainset, testset