import os
import shutil
import warnings
import requests
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/31
def getWSI_VQA(path):
    urls = [
        'https://raw.githubusercontent.com/cpystan/WSI-VQA/master/dataset/WSI_captions/WsiVQA_train.json',
        'https://raw.githubusercontent.com/cpystan/WSI-VQA/master/dataset/WSI_captions/WsiVQA_test.json',
        'https://raw.githubusercontent.com/cpystan/WSI-VQA/master/dataset/WSI_captions/WsiVQA_val.json',
        'https://raw.githubusercontent.com/cpystan/WSI-VQA/master/dataset/splits_0.csv',
    ]
    return datasetLoad(urls, path=path, datasetName="WSI_VQA")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(os.path.join(datasetPath,'data')):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            for url in urls:
                filename = url.split("/")[-1]
                if not os.path.exists(os.path.join(datasetPath, filename)):
                    download_file(url, os.path.join(datasetPath, filename))

            download_images(datasetPath)

            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def download_images(datasetPath):
    split_path = os.path.join(datasetPath, "splits_0.csv")
    split = pd.read_csv(split_path)
    train_imgs = (split["train"][~split["train"].isna()].apply(lambda x: x+".svs")).tolist()
    test_imgs = (split["test"][~split["test"].isna()].apply(lambda x: x+".svs")).tolist()
    val_imgs = (split["val"][~split["val"].isna()].apply(lambda x: x+".svs")).tolist()
    imgs = train_imgs + test_imgs + val_imgs

    url = "https://api.gdc.cancer.gov/files"
    params = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "=", 
                    "content": {
                        "field":"file_name",
                        "value":imgs
                    }
                }
            ]
        },
        "format": "JSON",
        "fields": "file_id",
        "size": 1000
    }

    response = requests.post(url, json=params)
    data = response.json()

    if 'data' in data and 'hits' in data['data']:
        if len(data['data']['hits']) > 0:
            uuids = [hit["id"] for hit in data['data']['hits']]
        else:
            print("file not found")
    else:
        raise AssertionError(f"error: {response.status_code}, error info: {data}")
    
    uuids_str = ",".join(uuids[:2])
    data_url = f"https://api.gdc.cancer.gov/data/{uuids_str}"
    downloadPath = os.path.join(datasetPath, "download.tar.gz")
    dataPath = os.path.join(datasetPath,'data')
    download_file(data_url, downloadPath, dataPath)

    for uuid in os.listdir(dataPath):
        uuidPath = os.path.join(dataPath, uuid)
        if os.path.isdir(uuidPath):
            for file_name in os.listdir(uuidPath):
                shutil.move(os.path.join(uuidPath, file_name), os.path.join(dataPath, file_name))
            shutil.rmtree(uuidPath)


def loadLocalFiles(path):
    dataPath = os.path.join(path, "data")
    df_train = pd.read_json(os.path.join(path,'WsiVQA_train.json'))
    df_test = pd.read_json(os.path.join(path,'WsiVQA_test.json'))
    df_val = pd.read_json(os.path.join(path,'WsiVQA_val.json'))

    svs_names = [fn for fn in os.listdir(dataPath) if ".svs" in fn]
    def check(x):
        result = []
        for svs in svs_names:
            if x in svs:
                result.append(os.path.join(dataPath, svs))
        return result
    
    df_train["svs_paths"] = df_train["Id"].apply(check)
    df_test["svs_paths"] = df_test["Id"].apply(check)
    df_val["svs_paths"] = df_val["Id"].apply(check)

    df_train = df_train[df_train["svs_paths"].apply(lambda x: x!=[])]
    df_test = df_test[df_test["svs_paths"].apply(lambda x: x!=[])]
    df_val = df_val[df_val["svs_paths"].apply(lambda x: x!=[])]

    trainset = df_train.to_dict(orient="records")
    testset = df_test.to_dict(orient="records")
    valset = df_val.to_dict(orient="records")
    
    return trainset, testset, valset