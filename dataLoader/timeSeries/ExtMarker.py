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

from ..utils import print_sys, download_file


def getExtMarker(path):
    urls = [
        'https://github.com/pohl-michel/time-series-forecasting-with-UORO-RTRL-LMS-and-linear-regression/archive/refs/heads/main.zip',
    ]
    return datasetLoad(urls, path=path, datasetName="ExtMarker")


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
            
            datafolder = os.path.join(datasetPath,"time-series-forecasting-with-UORO-RTRL-LMS-and-linear-regression-main","Original data")
            for fn in os.listdir(datafolder):
                shutil.move(os.path.join(datafolder,fn), datasetPath)

            shutil.rmtree(os.path.join(datasetPath,"time-series-forecasting-with-UORO-RTRL-LMS-and-linear-regression-main"))

            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    breakpoint()