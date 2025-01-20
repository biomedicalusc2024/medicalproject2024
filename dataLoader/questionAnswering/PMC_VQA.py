import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/19
def getPMC_VQA(path):
    urls = [
        "https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/images.zip?download=true",
        "https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/train.csv?download=true",
        "https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/test.csv?download=true",
    ]
    return datasetLoad(urls=urls, path=path, datasetName="PMC_VQA")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "raw.zip")
        train_path = os.path.join(datasetPath, "train.csv")
        test_path = os.path.join(datasetPath, "test.csv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            download_file(urls[0], datasetZip, datasetPath)
            download_file(urls[1], train_path)
            download_file(urls[2], test_path)
            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")

    img_path = os.path.join(path, "images")
    valid_list = os.listdir(img_path)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train = df_train[df_train['Figure_path'].isin(valid_list)]
    df_test = df_test[df_test['Figure_path'].isin(valid_list)]
    df_train['Figure_path'] = df_train['Figure_path'].apply(lambda x: os.path.join(img_path, x))
    df_test['Figure_path'] = df_test['Figure_path'].apply(lambda x: os.path.join(img_path, x))
    
    trainset = df_train.to_dict(orient='records')
    testset = df_test.to_dict(orient='records')
    return trainset, testset