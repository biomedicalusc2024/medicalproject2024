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


def getHoC(path):
    url = "https://github.com/sb895/Hallmarks-of-Cancer/archive/refs/heads/master.zip"
    return datasetLoad(url=url, path=path, datasetName="HoC")


def datasetLoad(url, path, datasetName):
    try:
        dataPath = os.path.join(path, datasetName)
        get_source(url, dataPath)
        return loadLocalFiles(dataPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    sourcePath = os.path.join(path, "text")
    targetPath = os.path.join(path, "labels")
    id_list = os.listdir(sourcePath)
    source_list = [[os.path.join(sourcePath, s)] for s in id_list]
    target_list = [[os.path.join(targetPath, s)] for s in id_list]
    return {
        "source": source_list,
        "target": target_list,
    }


def get_source(source_url, target_path):
    if not (os.path.exists(os.path.join(target_path, "labels")) and os.path.exists(os.path.join(target_path, "text"))):
        os.makedirs(target_path, 0o755, exist_ok=True)
        if os.path.exists(os.path.join(target_path, "Hallmarks-of-Cancer-master")):
            shutil.rmtree(os.path.join(target_path, "Hallmarks-of-Cancer-master"))
        zip_filepath = os.path.join(target_path,"master.zip")
        subprocess.call(['wget', '-P', target_path, source_url])
        try:
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(target_path)
        except Exception as e:
            print(f"An error occurred: {e}")
        os.remove(zip_filepath)
        shutil.move(os.path.join(target_path, "Hallmarks-of-Cancer-master", "labels"), target_path)
        shutil.move(os.path.join(target_path, "Hallmarks-of-Cancer-master", "text"), target_path)
        shutil.rmtree(os.path.join(target_path, "Hallmarks-of-Cancer-master"))