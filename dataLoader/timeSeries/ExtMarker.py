import os
import shutil
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/20
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
    fn_list = [fn for fn in os.listdir(path) if "csv" in fn]
    dataset = []
    for fn in fn_list:
        fn_full = os.path.join(path,fn)
        df = pd.read_csv(fn_full, sep=";")
        df["source"] = fn_full
        dataset += df.to_dict(orient='records')
    return dataset