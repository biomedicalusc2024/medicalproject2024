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
    columns = ['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position', 'OriginalImageWidth', 'OriginalImageHeight', 'OriginalImagePixelSpacingx', 'OriginalImagePixelSpacingy']
    source_columns = ['Image Index', 'Patient Age', 'Patient Gender', 'View Position', 'OriginalImageWidth', 'OriginalImageHeight', 'OriginalImagePixelSpacingx', 'OriginalImagePixelSpacingy']
    df_entry = pd.read_csv(f"{path}/Data_Entry_2017.csv", skiprows=1, names=columns, index_col='Image Index')
    train_val_dataset = {}
    test_dataset = {}

    with open(f"{path}/train_val_list.txt", 'r', encoding='utf-8') as file:
        content = file.read()
        train_val_df = df_entry.loc[content.split("\n")]
        train_val_dataset["target"] = train_val_df["Finding Labels"].apply(lambda x: x.split("|")).tolist()
        train_val_dataset["source"] = train_val_df.reset_index()[source_columns].to_numpy().tolist()
    
    with open(f"{path}/test_list.txt", 'r', encoding='utf-8') as file:
        content = file.read()
        test_df = df_entry.loc[content.split("\n")]
        test_dataset["target"] = test_df["Finding Labels"].apply(lambda x: x.split("|")).tolist()
        test_dataset["source"] = test_df.reset_index()[source_columns].to_numpy().tolist()

    return train_val_dataset, test_dataset
    
