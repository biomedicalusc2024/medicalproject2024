import os
import json
import warnings

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

MSD_SUBTITLE = ["Task01_BrainTumour", "Task02_Heart", "Task03_Liver", "Task04_Hippocampus",
                "Task05_Prostate", "Task06_Lung", "Task07_Pancreas", "Task08_HepaticVessel",
                "Task09_Spleen", "Task10_Colon"]


# tested by tjl 2025/1/24
def getMSD(path, subtitle):
    urls = {
        "Task01_BrainTumour": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
        "Task02_Heart": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
        "Task03_Liver": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
        "Task04_Hippocampus": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
        "Task05_Prostate": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
        "Task06_Lung": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
        "Task07_Pancreas": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
        "Task08_HepaticVessel": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
        "Task09_Spleen": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
        "Task10_Colon": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
    }

    if subtitle not in MSD_SUBTITLE:
        raise AttributeError(f"Please enter dataset name in MSD-subset format and select the subsection of MSD in {MSD_SUBTITLE}")
    return datasetLoad(urls=urls, subtitle=subtitle, path=path, datasetName="MSD")


def datasetLoad(urls, subtitle, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        taskPath = os.path.join(datasetPath, subtitle)
        tarPath = os.path.join(datasetPath, f"{subtitle}.tar")
        if os.path.exists(taskPath):
            print_sys("Found local copy...")
            return loadLocalFiles(taskPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[subtitle], tarPath, datasetPath)
            return loadLocalFiles(taskPath)
    except Exception as e:
        print_sys(f"error: {e}")

def loadLocalFiles(path):
    with open(os.path.join(path, "dataset.json"), 'r') as file:
        data_json = json.load(file)
    
    trainset = data_json["training"]
    testset = data_json["test"]

    trainset = [{k:os.path.join(path,v[2:]) for k,v in item.items()} for item in trainset]
    testset = [{"image":os.path.join(path,item[2:])} for item in testset]
    trainset = [item for item in trainset if (os.path.exists(item["image"]) and os.path.exists(item["label"]))]
    testset = [item for item in testset if (os.path.exists(item["image"]))]
    
    return trainset, testset