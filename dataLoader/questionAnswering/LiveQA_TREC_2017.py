import os
import shutil
import warnings
import pandas as pd
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/30
def getLiveQA_TREC_2017(path):
    # urls = ["https://github.com/abachaa/LiveQA_MedicalTask_TREC2017/archive/refs/heads/master.zip"]
    urls = [
        "https://huggingface.co/datasets/hyesunyun/liveqa_medical_trec2017/resolve/main/TREC2017-LiveQA-Medical-Train1.jsonl?download=true",
        "https://huggingface.co/datasets/hyesunyun/liveqa_medical_trec2017/resolve/main/TREC2017-LiveQA-Medical-Train2.jsonl?download=true",
        "https://huggingface.co/datasets/hyesunyun/liveqa_medical_trec2017/resolve/main/test.jsonl?download=true"
    ]
    return datasetLoad(urls=urls, path=path, datasetName="LiveQA_TREC_2017")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], os.path.join(datasetPath,"TREC2017-LiveQA-Medical-Train1.jsonl"))
            download_file(urls[1], os.path.join(datasetPath,"TREC2017-LiveQA-Medical-Train2.jsonl"))
            download_file(urls[2], os.path.join(datasetPath,"test.jsonl"))
            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    # train1 = pd.read_json(os.path.join(path, "TREC2017-LiveQA-Medical-Train1.jsonl"), lines=True)
    # train2 = pd.read_json(os.path.join(path, "TREC2017-LiveQA-Medical-Train2.jsonl"), lines=True)
    test = pd.read_json(os.path.join(path, "test.jsonl"), lines=True)

    # trainset = train1.to_dict(orient="records") + train2.to_dict(orient="records")
    testset = test.to_dict(orient="records")
    
    return testset

