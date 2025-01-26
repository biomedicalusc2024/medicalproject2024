import os
import warnings

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/23
def getLA(path):
    urls = ["https://www.dropbox.com/scl/fi/nero2nlaocdcdfhzwu5h0/2018_UTAH_MICCAI.zip?rlkey=vkkfrkc2l6x1e61jqyutb35qn&e=3&st=9tzk7mgs&dl=1"]
    return datasetLoad(urls=urls, path=path, datasetName="LA")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        zipPath = os.path.join(datasetPath, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], zipPath, datasetPath)
            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    trainPath = os.path.join(path, "Training Set")
    testPath = os.path.join(path, "Testing Set")

    trainset = []
    for id in os.listdir(trainPath):
        ele = {"id":id}
        id_path = os.path.join(trainPath, id)
        for fn in os.listdir(id_path):
            if ".nrrd" in fn:
                fid = fn.split(".")[0]
                fn_full = os.path.join(id_path, fn)
                ele[fid] = fn_full
        trainset.append(ele)

    testset = []
    for id in os.listdir(testPath):
        ele = {"id":id}
        id_path = os.path.join(testPath, id)
        for fn in os.listdir(id_path):
            fid = fn.split(".")[0]
            fn_full = os.path.join(id_path, fn)
            ele[fid] = fn_full
        testset.append(ele)
    
    return trainset, testset