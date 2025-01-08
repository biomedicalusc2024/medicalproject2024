import os
import shutil
import zipfile
import requests
import warnings
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys


# no training set left, only test set


def getLiveQA_PREC_2017(path):
    url = "https://github.com/abachaa/LiveQA_MedicalTask_TREC2017/archive/refs/heads/master.zip"
    return datasetLoad(url=url, path=path, datasetName="LiveQA_PREC_2017")


def datasetLoad(url, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            print_sys("Downloading...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
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

                tempdir = os.path.join(datasetPath, "LiveQA_MedicalTask_TREC2017-master", "TestDataset")
                for filename in os.listdir(tempdir):
                    shutil.move(os.path.join(tempdir, filename), datasetPath)
                shutil.rmtree(os.path.join(datasetPath, "LiveQA_MedicalTask_TREC2017-master"))

                return loadLocalFiles(datasetPath)
            else:
                print_sys("Connection error, please check the internet.")
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    breakpoint()
    sourcePath = os.path.join(path, "text")
    targetPath = os.path.join(path, "labels")
    id_list = os.listdir(sourcePath)
    source_list = [[os.path.join(sourcePath, s)] for s in id_list]
    target_list = [[os.path.join(targetPath, s)] for s in id_list]
    return {
        "source": source_list,
        "target": target_list,
    }

