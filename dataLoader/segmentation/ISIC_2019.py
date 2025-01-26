import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/23
def getISIC_2019(path):
    urls = [
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv",
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Metadata.csv",
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_GroundTruth.csv",
    ]
    return datasetLoad(urls=urls, path=path, datasetName="ISIC_2019")


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
                downloadFilePath = os.path.join(datasetPath,fn)
                if "csv" in fn:
                    download_file(url, downloadFilePath)
                else:
                    download_file(url, downloadFilePath, datasetPath)

            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    trainInputPath = os.path.join(path, "ISIC_2019_Training_Input")
    trainInputs = [fn for fn in os.listdir(trainInputPath) if ".jpg" in fn]
    testInputPath = os.path.join(path, "ISIC_2019_Test_Input")
    testInputs = [fn for fn in os.listdir(testInputPath) if ".jpg" in fn]
    trainMeta = pd.read_csv(os.path.join(path, "ISIC_2019_Training_Metadata.csv")).set_index("image")
    testMeta = pd.read_csv(os.path.join(path, "ISIC_2019_Test_Metadata.csv")).set_index("image")
    trainGroundtruth = pd.read_csv(os.path.join(path, "ISIC_2019_Training_GroundTruth.csv")).set_index("image")
    testGroundtruth = pd.read_csv(os.path.join(path, "ISIC_2019_Test_GroundTruth.csv")).set_index("image")

    train_df = trainMeta.merge(trainGroundtruth, left_index=True, right_index=True, how='inner')
    test_df = testMeta.merge(testGroundtruth, left_index=True, right_index=True, how='inner')

    train_df = train_df.reset_index()
    test_df = test_df.reset_index()

    train_df["image"] = train_df["image"].apply(lambda x: f"{x}.jpg")
    test_df["image"] = test_df["image"].apply(lambda x: f"{x}.jpg")

    train_df = train_df[train_df["image"].isin(trainInputs)]
    test_df = test_df[test_df["image"].isin(testInputs)]

    train_df["image"] = train_df["image"].apply(lambda x: os.path.join(trainInputPath, x))
    test_df["image"] = test_df["image"].apply(lambda x: os.path.join(testInputPath, x))

    trainset = train_df.to_dict(orient='records')
    testset = test_df.to_dict(orient='records')
    return trainset, testset