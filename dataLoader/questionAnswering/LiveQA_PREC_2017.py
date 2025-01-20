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

from ..utils import print_sys, download_file


# need guidance on data needed
def getLiveQA_PREC_2017(path):
    urls = ["https://github.com/abachaa/LiveQA_MedicalTask_TREC2017/archive/refs/heads/master.zip"]
    return datasetLoad(urls=urls, path=path, datasetName="LiveQA_PREC_2017")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            download_file(urls[0], datasetZip, datasetPath)
            tempdir = os.path.join(datasetPath, "LiveQA_MedicalTask_TREC2017-master", "TestDataset")
            for filename in os.listdir(tempdir):
                shutil.move(os.path.join(tempdir, filename), datasetPath)
            shutil.rmtree(os.path.join(datasetPath, "LiveQA_MedicalTask_TREC2017-master"))

            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    breakpoint()

