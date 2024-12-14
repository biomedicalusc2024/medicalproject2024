import os
import shutil
import requests
import zipfile
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getChestXRays(path):
    url = "https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data"
    return datasetLoad(url=url, path=path, datasetName="ChestXRays")


def datasetLoad(url, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetImagePath = os.path.join(datasetPath, "images")
        datasetZip = os.path.join(datasetPath, "raw.zip")
        if os.path.exists(datasetImagePath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            print_sys("Downloading...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                os.makedirs(datasetPath, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(datasetZip, "wb") as file:
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

                with zipfile.ZipFile(datasetZip, "r") as z:
                    z.extractall(datasetPath)
                print_sys("Extraction complete.")
                os.remove(datasetZip)

                entries = os.listdir(datasetPath)
                subdirectories = [os.path.join(datasetPath,entry) for entry in entries if os.path.isdir(os.path.join(datasetPath,entry)) and entry!="images"]
                for subdir in subdirectories:
                    subdir_image = os.path.join(subdir,"images")
                    for file in os.listdir(subdir_image):
                        src_file_path = os.path.join(subdir_image, file)
                        shutil.move(src_file_path, datasetImagePath)
                    shutil.rmtree(subdir)

                return loadLocalFiles(datasetPath)
            else:
                print_sys("Connection error, please check the internet.")
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df_box = pd.read_csv(f"{path}/BBox_List_2017.csv", header=None, skiprows=1)
    source, target = [], []
    for _, row in df_box.iterrows():
        file_path = f"{path}/images/{row[0]}"
        box = [row[2], row[3], row[4], row[5]]
        source.append(file_path)
        target.append(box)
    return {
        "source": source,
        "target": target,
    }
    
