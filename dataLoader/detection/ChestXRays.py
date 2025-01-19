import os
import shutil
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/18
def getChestXRays(path):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data"]
    return datasetLoad(urls=urls, path=path, datasetName="ChestXRays")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetImagePath = os.path.join(datasetPath, "images")
        datasetZip = os.path.join(datasetPath, "raw.zip")
        if os.path.exists(datasetImagePath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], datasetZip, datasetPath)
            entries = os.listdir(datasetPath)
            subdirectories = [os.path.join(datasetPath,entry) for entry in entries if os.path.isdir(os.path.join(datasetPath,entry)) and entry!="images"]
            for subdir in subdirectories:
                subdir_image = os.path.join(subdir,"images")
                for file in os.listdir(subdir_image):
                    src_file_path = os.path.join(subdir_image, file)
                    shutil.move(src_file_path, datasetImagePath)
                shutil.rmtree(subdir)
            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df_box = pd.read_csv(f"{path}/BBox_List_2017.csv", header=None, skiprows=1, names=["img_path", "Finding Label", "Box_x", "Box_y", "Box_w", "Box_h"])
    valid_list = os.listdir(os.path.join(path, "images"))
    df_box = df_box[df_box['img_path'].isin(valid_list)]
    df_box["img_path"] = df_box["img_path"].apply(lambda x: path+"/images/"+x)
    dataset = df_box.to_dict(orient='records')
    return dataset