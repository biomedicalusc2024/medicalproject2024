import os
import json
import warnings

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/24
def getHippo(path):
    urls = ["https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"] 
    return datasetLoad(urls=urls, path=path, datasetName="Hippo")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        rarPath = os.path.join(datasetPath,'raw.tar')
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], rarPath, datasetPath)
            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    basePath = os.path.join(path, "Task04_Hippocampus")
    with open(os.path.join(path, "Task04_Hippocampus", "dataset.json"), 'r') as file:
        data_json = json.load(file)
    
    trainset = data_json["training"]
    testset = data_json["test"]

    trainset = [{k:os.path.join(basePath,v[2:]) for k,v in item.items()} for item in trainset]
    testset = [{"image":os.path.join(basePath,item[2:])} for item in testset]
    trainset = [item for item in trainset if (os.path.exists(item["image"]) and os.path.exists(item["label"]))]
    testset = [item for item in testset if (os.path.exists(item["image"]))]
    
    return trainset, testset