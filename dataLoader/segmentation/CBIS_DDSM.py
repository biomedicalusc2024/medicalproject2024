import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/25
def getCBIS_DDSM(path):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/awsaf49/cbis-ddsm-breast-cancer-image-dataset"]
    return datasetLoad(urls=urls, path=path, datasetName="CBIS_DDSM")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        zipPath = os.path.join(datasetPath,'raw.zip')
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
    csvPath, jpegPath = os.path.join(path, "csv"), os.path.join(path, "jpeg")
    # cal_train_df = pd.read_csv(os.path.join(csvPath, "calc_case_description_train_set.csv"))
    # cal_test_df = pd.read_csv(os.path.join(csvPath, "calc_case_description_test_set.csv"))
    # mass_train_df = pd.read_csv(os.path.join(csvPath, "mass_case_description_train_set.csv"))
    # mass_test_df = pd.read_csv(os.path.join(csvPath, "mass_case_description_test_set.csv"))
    dicom_df = pd.read_csv(os.path.join(csvPath, "dicom_info.csv"))
    dicom_df["image_path"] = dicom_df["image_path"].apply(lambda x: x.replace('CBIS-DDSM/jpeg', jpegPath))

    dataset = dicom_df.to_dict(orient='records')
    return dataset
    