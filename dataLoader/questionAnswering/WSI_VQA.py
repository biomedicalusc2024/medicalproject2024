import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# img resource not provided, need to download manually.
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


def loadLocalFiles(path):
    df_train = pd.read_json(os.path.join(path,'train.json'))
    df_test = pd.read_json(os.path.join(path,'test.json'))
    df_val = pd.read_json(os.path.join(path,'val.json'))
    breakpoint()