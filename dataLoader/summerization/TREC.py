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
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

from ..utils import print_sys


def getTREC(path):
    urls = [
        'https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part1.zip',
        'https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part2.zip',
        'https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part3.zip',
        'https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part4.zip',
        'https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part5.zip',
    ]
    return datasetLoad(urls, path=path, datasetName="TREC")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            for url in urls:
                fn = url.split("/")[-1]
                print('downloading '+fn+'...')
                download_file(url, os.path.join(datasetPath, fn), datasetPath)
            
            for subdir in os.listdir(datasetPath):
                for f in os.listdir(os.path.join(datasetPath, subdir)):
                    shutil.move(os.path.join(datasetPath, subdir, f), os.path.join(datasetPath, f))
                shutil.rmtree(os.path.join(datasetPath, subdir))

            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    breakpoint()


def download_file(url, destination, extractionPath=None):
    try:
        headers = {
            'Accept': '*/*',
             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        response = requests.get(url, stream=True, headers=headers)
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

        else:
            print_sys("url error")
    except Exception as e:
        print_sys(f"error: {e}")