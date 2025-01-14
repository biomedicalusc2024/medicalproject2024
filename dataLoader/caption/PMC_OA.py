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

def getPMC_OA(path):
    urls = [
        'https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/pmc_oa.jsonl?download=true',
        'https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/images.zip?download=true',
    ]
    return datasetLoad(urls, path=path, datasetName="PMC_OA")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        imgPath = os.path.join(datasetPath,'images')
        if os.path.exists(imgPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(os.path.join(datasetPath,'images'), exist_ok=True)

            download_file(urls[0], os.path.join(datasetPath, "pmc_oa.jsonl"))
            download_file(urls[1], os.path.join(datasetPath, "images.zip"), imgPath)

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
    df = pd.read_json(os.path.join(path, "pmc_oa.jsonl"), lines=True)
    img_folder = os.path.join(path, "images", "caption_T060_filtered_top4_sep_v0_subfigures")
    imgs = os.listdir(img_folder)
    df = df[df['image'].isin(imgs)]
    source_cols = ['image', 'alignment_type', 'alignment_score']
    target_cols = ['caption']
    df["image"] = df["image"].apply(lambda x: img_folder+"/"+x)
    return {
        "source": df[source_cols].to_numpy().tolist(),
        "target": df[target_cols].to_numpy().tolist(),
    }