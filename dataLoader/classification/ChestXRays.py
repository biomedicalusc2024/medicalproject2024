import os
import shutil
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/21
def getChestXRays(path):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data"]
    return datasetLoad(urls=urls, path=path, datasetName="ChestXRays")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetImagePath = os.path.join(datasetPath, "images")
        datasetZip = os.path.join(datasetPath, "raw.zip")
        if os.path.exists(datasetImagePath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], datasetZip, datasetPath)
            
            entries = os.listdir(datasetPath)
            subdirectories = [os.path.join(datasetPath,entry) for entry in entries if os.path.isdir(os.path.join(datasetPath,entry)) and entry!="images"]
            for subdir in subdirectories:
                subdir_image = os.path.join(subdir,"images")
                for file in os.listdir(subdir_image):
                    src_file_path = os.path.join(subdir_image, file)
                    shutil.move(src_file_path, datasetImagePath)
                shutil.rmtree(subdir)

            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    columns = ['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position', 'OriginalImageWidth', 'OriginalImageHeight', 'OriginalImagePixelSpacingx', 'OriginalImagePixelSpacingy']
    df_entry = pd.read_csv(os.path.join(path, "Data_Entry_2017.csv"), names=columns, skiprows=1)

    with open(os.path.join(path, "train_val_list.txt"), 'r') as file:
        train_split = file.readlines()
        train_split = [l.strip() for l in train_split]
    with open(os.path.join(path, "test_list.txt"), 'r') as file:
        test_split = file.readlines()
        test_split = [l.strip() for l in test_split]

    imgPath = os.path.join(path, "images")
    imgs = os.listdir(imgPath)
    
    train_imgs = list(set(imgs) & set(train_split))
    test_imgs = list(set(imgs) & set(test_split))

    df_train = df_entry[df_entry['Image Index'].isin(train_imgs)]
    df_test = df_entry[df_entry['Image Index'].isin(test_imgs)]

    df_train['Image Index'] = df_train['Image Index'].apply(lambda x: os.path.join(imgPath, x))
    df_test['Image Index'] = df_test['Image Index'].apply(lambda x: os.path.join(imgPath, x))
    
    trainset = df_train.to_dict(orient='records')
    testset = df_test.to_dict(orient='records')


    return trainset, testset
    
