import os
import json
import shutil
import rarfile
import tarfile
import zipfile
import requests
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getWSI_VQA(path):
    urls = [
        'https://raw.githubusercontent.com/cpystan/WSI-VQA/master/dataset/WSI_captions/WsiVQA_train.json',
        'https://raw.githubusercontent.com/cpystan/WSI-VQA/master/dataset/WSI_captions/WsiVQA_test.json',
        'https://raw.githubusercontent.com/cpystan/WSI-VQA/master/dataset/WSI_captions/WsiVQA_val.json',
    ]
    return datasetLoad(urls, path=path, datasetName="WSI_VQA")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(os.path.join(datasetPath,'data')):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(os.path.join(datasetPath,'data'), exist_ok=True)

            for url in urls:
                filename = url.split("_")[-1]
                download_file(url, os.path.join(datasetPath, filename))

            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def download_file(url, destination, extractionPath=None):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(destination, "wb") as file:
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

            if extractionPath:
                if "zip" in destination:
                    with zipfile.ZipFile(destination, "r") as z:
                        z.extractall(extractionPath)
                    print_sys("Extraction complete.")
                    os.remove(destination)
                elif "rar" in destination:
                    with rarfile.RarFile(destination) as rf:
                        rf.extractall(extractionPath)
                    print_sys("Extraction complete.")
                    os.remove(destination)
                elif "tar" in destination:
                    if "gz" in destination:
                        with tarfile.open(destination, 'r:gz') as tar:
                            tar.extractall(extractionPath)
                        print_sys("Extraction complete.")
                        os.remove(destination)
                    else:
                        with tarfile.open(destination, 'r') as tar:
                            tar.extractall(extractionPath)
                        print_sys("Extraction complete.")
                        os.remove(destination)

    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df_train = pd.read_json(os.path.join(path,'train.json'))
    df_test = pd.read_json(os.path.join(path,'test.json'))
    df_val = pd.read_json(os.path.join(path,'val.json'))
    breakpoint()