import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/20
def getChestX_ray8(path):
    urls = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz',
        "https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data/test_list.txt",
        "https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data/train_val_list.txt",
        "https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data/Data_Entry_2017.csv",
    ]
    return datasetLoad(urls, path=path, datasetName="ChestX_ray8")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        imgPath = os.path.join(datasetPath, "images")
        if os.path.exists(imgPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)

            for idx, url in enumerate(urls[:12]):
                fn = 'images_%02d.tar.gz' % (idx+1)
                print('downloading'+fn+'...')
                download_file(url, os.path.join(datasetPath, fn), datasetPath)
            for url in urls[12:13]:
                fn = url.split("/")[-1]
                download_file(url, os.path.join(datasetPath, fn))
            for url in urls[13:]:
                fn = url.split("/")[-1] + ".zip"
                download_file(url, os.path.join(datasetPath, fn), datasetPath)
            
            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    columns = ['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position', 
               'OriginalImage-Width', 'OriginalImage-Height', 'OriginalImagePixelSpacing-x', 'OriginalImagePixelSpacing-y']
    dataEntry = pd.read_csv(os.path.join(path, "Data_Entry_2017.csv"), names=columns, skiprows=1)
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

    df_train = dataEntry[dataEntry['Image Index'].isin(train_imgs)]
    df_test = dataEntry[dataEntry['Image Index'].isin(test_imgs)]

    df_train['Image Index'] = df_train['Image Index'].apply(lambda x: os.path.join(imgPath, x))
    df_test['Image Index'] = df_test['Image Index'].apply(lambda x: os.path.join(imgPath, x))
    
    trainset = df_train.to_dict(orient='records')
    testset = df_test.to_dict(orient='records')

    return trainset, testset