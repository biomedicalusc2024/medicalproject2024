import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

MedQA_USMLE_SUBTITLE = ["Mainland", "Taiwan", "US", "all"]


# tested by tjl 2025/1/19
def getMedQA_USMLE(path, subtitle):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/moaaztameer/medqa-usmle"]
    return datasetLoad(urls=urls, subtitle=subtitle, path=path, datasetName="MedQA_USMLE")


def datasetLoad(urls, subtitle, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath, subtitle)
        else:
            download_file(urls[0], datasetZip, datasetPath)
            return loadLocalFiles(datasetPath, subtitle)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path, subtitle):
    basePath = os.path.join(path, "MedQA-USMLE", "questions")
    if subtitle == "Mainland":
        basePath = os.path.join(basePath, "Mainland")
    elif subtitle == "Taiwan":
        basePath = os.path.join(basePath, "Taiwan")
    elif subtitle == "US":
        basePath = os.path.join(basePath, "US")
    elif subtitle == "all":
        trainset, testset, valset = [], [], []
        for sub_path in ["Mainland", "Taiwan", "US"]:
            subPath = os.path.join(basePath, sub_path)
            df_train = pd.read_json(os.path.join(subPath, "train.jsonl"), lines=True)
            df_test = pd.read_json(os.path.join(subPath, "test.jsonl"), lines=True)
            df_val = pd.read_json(os.path.join(subPath, "dev.jsonl"), lines=True)


            trainset = trainset + df_train.to_dict(orient='records')
            testset = testset + df_test.to_dict(orient='records')
            valset = valset + df_val.to_dict(orient='records')
            
        return trainset, testset, valset
    else:
        raise AttributeError(f'Please enter dataset name in MedQA_USMLE-subset format and select the subsection of MedQA_USMLE in {MedQA_USMLE_SUBTITLE}')
    
    df_train = pd.read_json(os.path.join(basePath, "train.jsonl"), lines=True)
    df_test = pd.read_json(os.path.join(basePath, "test.jsonl"), lines=True)
    df_val = pd.read_json(os.path.join(basePath, "dev.jsonl"), lines=True)

    trainset = df_train.to_dict(orient='records')
    testset = df_test.to_dict(orient='records')
    valset = df_val.to_dict(orient='records')

    return trainset, testset, valset