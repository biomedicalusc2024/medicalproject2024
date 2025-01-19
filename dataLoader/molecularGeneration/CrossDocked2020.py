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

from ..utils import print_sys, download_file


def getCrossDocked2020(path):
    # urls = [
    #     "http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3.tgz",
    #     "http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3_types.tgz",
    # ]

    # splits = ["test0", "test1", "test2", "train0", "train1", "train2"]
    # typesNames = []
    # typesNames += [f"cdonly_it2_tt_v1.3_0_{x}.types" for x in splits]
    # typesNames += [f"it0_tt_v1.3_0_{x}.types" for x in splits]
    # typesNames += [f"it2_redocked_tt_v1.3_0_{x}.types" for x in splits]
    # typesNames += [f"it2_redocked_tt_v1.3_completeset_{x}.types" for x in ["test0", "train0"]]
    # typesNames += [f"it2_tt_v1.3_0_{x}.types" for x in splits]
    # typesNames += [f"it2_tt_v1.3_10p20n_{x}.types" for x in splits]
    # typesNames += [f"it2_tt_v1.3_completeset_{x}.types" for x in ["test0", "train0"]]
    # typesNames += [f"mod_it2_tt_v1.3_0_{x}.types" for x in splits]
    # typesNames += [f"mod_it2_tt_v1.3_completeset_{x}.types" for x in ["test0", "train0"]]

    urls = [
        "http://bits.csb.pitt.edu/files/crossdock2020/downsampled_CrossDocked2020_v1.3.tgz",
        "http://bits.csb.pitt.edu/files/crossdock2020/downsampled_CrossDocked2020_v1.3_types.tgz",
    ]

    splits = ["test0", "test1", "test2", "train0", "train1", "train2"]
    typesNames = []
    typesNames += [f"it2_tt_v1.3_10p20n_{x}.types" for x in splits]
    
    return datasetLoad(urls, path=path, datasetName="CrossDocked2020")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        dockerPath = os.path.join(datasetPath, "dockers")
        typesPath = os.path.join(datasetPath, "types")
        if (os.path.exists(dockerPath) and os.path.exists(typesPath)):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath)
            
            fn = urls[0].split("/")[-1]
            download_file(urls[0], os.path.join(datasetPath, fn), os.path.join(datasetPath, "dockers"))

            fn = urls[1].split("/")[-1]
            download_file(urls[1], os.path.join(datasetPath, fn), os.path.join(datasetPath))

            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    breakpoint()
    df_train = pd.read_csv(os.path.join(path, "train.csv"))
    df_test = pd.read_csv(os.path.join(path, "test.csv"))

    source_cols = ["SMILES"]
    target_cols = []

    trainset = {
        "source": df_train[source_cols].to_numpy().tolist(),
        "target": df_train[target_cols].to_numpy().tolist(),
    }
    testset = {
        "source": df_test[source_cols].to_numpy().tolist(),
        "target": df_test[target_cols].to_numpy().tolist(),
    }
    
    return trainset, testset
    