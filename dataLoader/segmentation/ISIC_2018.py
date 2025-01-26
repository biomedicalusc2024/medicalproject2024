import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

ISIC_2018_SUBTITLE = ["Task1", "Task2", "Task3"]


# tested by tjl 2025/1/23
def getISIC_2018(path, subtitle):
    urls = [
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Training_GroundTruth_v3.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Validation_GroundTruth.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Test_GroundTruth.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Test_GroundTruth.zip",

        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_LesionGroupings.csv",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_GroundTruth.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_GroundTruth.zip",
    ]
    if subtitle not in ISIC_2018_SUBTITLE:
        raise AttributeError(f"Please enter dataset name in ISIC_2018-subset format and select the subsection of ISIC_2018 in {ISIC_2018_SUBTITLE}")
    return datasetLoad(urls=urls, subtitle=subtitle, path=path, datasetName="ISIC_2018")


def datasetLoad(urls, subtitle, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath, subtitle)
        else:
            os.makedirs(datasetPath, exist_ok=True)

            for url in urls:
                fn = url.split("/")[-1]
                downloadFilePath = os.path.join(datasetPath,fn)
                if "csv" in fn:
                    download_file(url, downloadFilePath)
                else:
                    download_file(url, downloadFilePath, datasetPath)

            return loadLocalFiles(datasetPath, subtitle)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path, subtitle):
    if subtitle == "Task1":
        train_input_path = "ISIC2018_Task1-2_Training_Input"
        train_groundtruth_path = "ISIC2018_Task1_Training_GroundTruth"
        test_input_path = "ISIC2018_Task1-2_Test_Input"
        test_groundtruth_path = "ISIC2018_Task1_Test_GroundTruth"
        val_input_path = "ISIC2018_Task1-2_Validation_Input"
        val_groundtruth_path = "ISIC2018_Task1_Validation_GroundTruth"

        train_input_path = os.path.join(path, train_input_path)
        train_groundtruth_path = os.path.join(path, train_groundtruth_path)
        test_input_path = os.path.join(path, test_input_path)
        test_groundtruth_path = os.path.join(path, test_groundtruth_path)
        val_input_path = os.path.join(path, val_input_path)
        val_groundtruth_path = os.path.join(path, val_groundtruth_path)

        trainset, testset, valset = [], [], []

        train_groundtruth_fns = [fn for fn in os.listdir(train_groundtruth_path) if ".png" in fn]
        for fn in os.listdir(train_input_path):
            if ".jpg" in fn:
                fid = fn.split(".")[0]
                input_path = os.path.join(train_input_path, fn)
                groundtruths = [os.path.join(train_groundtruth_path, fn) for fn in train_groundtruth_fns if fid in fn]
                trainset.append({"input_path":input_path,"groundtruth_path":groundtruths})
        
        test_groundtruth_fns = [fn for fn in os.listdir(test_groundtruth_path) if ".png" in fn]
        for fn in os.listdir(test_input_path):
            if ".jpg" in fn:
                fid = fn.split(".")[0]
                input_path = os.path.join(test_input_path, fn)
                groundtruths = [os.path.join(test_groundtruth_path, fn) for fn in test_groundtruth_fns if fid in fn]
                testset.append({"input_path":input_path,"groundtruth_path":groundtruths})

        val_groundtruth_fns = [fn for fn in os.listdir(val_groundtruth_path) if ".png" in fn]
        for fn in os.listdir(val_input_path):
            if ".jpg" in fn:
                fid = fn.split(".")[0]
                input_path = os.path.join(val_input_path, fn)
                groundtruths = [os.path.join(val_groundtruth_path, fn) for fn in val_groundtruth_fns if fid in fn]
                valset.append({"input_path":input_path,"groundtruth_path":groundtruths})

    elif subtitle == "Task2":
        train_input_path = "ISIC2018_Task1-2_Training_Input"
        train_groundtruth_path = "ISIC2018_Task2_Training_GroundTruth_v3"
        test_input_path = "ISIC2018_Task1-2_Test_Input"
        test_groundtruth_path = "ISIC2018_Task2_Test_GroundTruth"
        val_input_path = "ISIC2018_Task1-2_Validation_Input"
        val_groundtruth_path = "ISIC2018_Task2_Validation_GroundTruth"

        train_input_path = os.path.join(path, train_input_path)
        train_groundtruth_path = os.path.join(path, train_groundtruth_path)
        test_input_path = os.path.join(path, test_input_path)
        test_groundtruth_path = os.path.join(path, test_groundtruth_path)
        val_input_path = os.path.join(path, val_input_path)
        val_groundtruth_path = os.path.join(path, val_groundtruth_path)

        trainset, testset, valset = [], [], []

        train_groundtruth_fns = [fn for fn in os.listdir(train_groundtruth_path) if ".png" in fn]
        for fn in os.listdir(train_input_path):
            if ".jpg" in fn:
                fid = fn.split(".")[0]
                input_path = os.path.join(train_input_path, fn)
                groundtruths = [os.path.join(train_groundtruth_path, fn) for fn in train_groundtruth_fns if fid in fn]
                trainset.append({"input_path":input_path,"groundtruth_path":groundtruths})
        
        test_groundtruth_fns = [fn for fn in os.listdir(test_groundtruth_path) if ".png" in fn]
        for fn in os.listdir(test_input_path):
            if ".jpg" in fn:
                fid = fn.split(".")[0]
                input_path = os.path.join(test_input_path, fn)
                groundtruths = [os.path.join(test_groundtruth_path, fn) for fn in test_groundtruth_fns if fid in fn]
                testset.append({"input_path":input_path,"groundtruth_path":groundtruths})

        val_groundtruth_fns = [fn for fn in os.listdir(val_groundtruth_path) if ".png" in fn]
        for fn in os.listdir(val_input_path):
            if ".jpg" in fn:
                fid = fn.split(".")[0]
                input_path = os.path.join(val_input_path, fn)
                groundtruths = [os.path.join(val_groundtruth_path, fn) for fn in val_groundtruth_fns if fid in fn]
                valset.append({"input_path":input_path,"groundtruth_path":groundtruths})
                
    elif subtitle == "Task3":
        train_input_path = "ISIC2018_Task3_Training_Input"
        train_groundtruth_path = "ISIC2018_Task3_Training_GroundTruth"
        test_input_path = "ISIC2018_Task3_Test_Input"
        test_groundtruth_path = "ISIC2018_Task3_Test_GroundTruth"
        val_input_path = "ISIC2018_Task3_Validation_Input"
        val_groundtruth_path = "ISIC2018_Task3_Validation_GroundTruth"

        train_input_path = os.path.join(path, train_input_path)
        train_groundtruth_path = os.path.join(path, train_groundtruth_path)
        test_input_path = os.path.join(path, test_input_path)
        test_groundtruth_path = os.path.join(path, test_groundtruth_path)
        val_input_path = os.path.join(path, val_input_path)
        val_groundtruth_path = os.path.join(path, val_groundtruth_path)

        train_input_fns = [fn for fn in os.listdir(train_input_path) if ".jpg" in fn]
        train_df = pd.read_csv(os.path.join(train_groundtruth_path, "ISIC2018_Task3_Training_GroundTruth.csv"))
        train_df["image"] = train_df["image"].apply(lambda x: x+".jpg")
        train_df = train_df[train_df["image"].isin(train_input_fns)]
        train_df["image"] = train_df["image"].apply(lambda x: os.path.join(train_input_path,x))
        trainset = train_df.to_dict(orient="records")
        
        test_input_fns = [fn for fn in os.listdir(test_input_path) if ".jpg" in fn]
        test_df = pd.read_csv(os.path.join(test_groundtruth_path, "ISIC2018_Task3_Test_GroundTruth.csv"))
        test_df["image"] = test_df["image"].apply(lambda x: x+".jpg")
        test_df = test_df[test_df["image"].isin(test_input_fns)]
        test_df["image"] = test_df["image"].apply(lambda x: os.path.join(test_input_path,x))
        testset = test_df.to_dict(orient="records")

        val_input_fns = [fn for fn in os.listdir(val_input_path) if ".jpg" in fn]
        val_df = pd.read_csv(os.path.join(val_groundtruth_path, "ISIC2018_Task3_Validation_GroundTruth.csv"))
        val_df["image"] = val_df["image"].apply(lambda x: x+".jpg")
        val_df = val_df[val_df["image"].isin(val_input_fns)]
        val_df["image"] = val_df["image"].apply(lambda x: os.path.join(val_input_path,x))
        valset = val_df.to_dict(orient="records")

    return trainset, testset, valset