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


def getChestX_ray8(path):
    urls = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz',
    ]
    return datasetLoad(urls, path=path, datasetName="ChestX_ray8")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        imgPath = os.path.join(datasetPath, "images")
        if os.path.exists(imgPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            for idx, url in enumerate(urls):
                fn = 'images_%02d.tar.gz' % (idx+1)
                print('downloading'+fn+'...')
                download_file(url, os.path.join(datasetPath, fn), datasetPath)

            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    columns = ['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position', 
               'OriginalImage-Width', 'OriginalImage-Height', 'OriginalImagePixelSpacing-x', 'OriginalImagePixelSpacing-y']
    source_cols = ['Image Index', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position', 
               'OriginalImage-Width', 'OriginalImage-Height', 'OriginalImagePixelSpacing-x', 'OriginalImagePixelSpacing-y']
    target_cols = ['Finding Labels']
    dataEntry = pd.read_csv(os.path.join(path, "Data_Entry_2017_v2020.csv"), names=columns, skiprows=1)
    with open(os.path.join(path, "train_val_list.txt"), 'r') as file:
        train_split = file.readlines()
        train_split = [l.strip() for l in train_split]
    with open(os.path.join(path, "test_list.txt"), 'r') as file:
        test_split = file.readlines()
        test_split = [l.strip() for l in test_split]
    imgPath = os.path.join(path, "images")
    imgs = os.listdir(imgPath)
    train_imgs = list(set(imgs) & set(train_split))
    test_imgs = list(set(imgs) & set(test_split))
    df_train = dataEntry[dataEntry['Image Index'].isin(train_imgs)]
    df_test = dataEntry[dataEntry['Image Index'].isin(test_imgs)]
    
    trainset = {
        "source": df_train[source_cols].to_numpy().tolist(),
        "target": df_train[target_cols].to_numpy().tolist(),
    }

    testset = {
        "source": df_test[source_cols].to_numpy().tolist(),
        "target": df_test[target_cols].to_numpy().tolist(),
    }

    return trainset, testset


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