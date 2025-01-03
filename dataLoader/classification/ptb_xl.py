import os
import ast
import wfdb
import shutil
import zipfile
import requests
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getPTB_XL(path):
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    return datasetLoad(url=url, path=path, datasetName="PTB-XL")


def datasetLoad(url, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            print_sys("Downloading...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))

                with open(datasetZip, "wb") as file:
                    if total_size == 0:
                        pbar = None
                    else:
                        pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))
                    if pbar:
                        pbar.close()
                print_sys("Download complete.")

                with zipfile.ZipFile(datasetZip, "r") as z:
                    z.extractall(datasetPath)
                print_sys("Extraction complete.")
                os.remove(datasetZip)

                return loadLocalFiles(datasetPath)
            else:
                print_sys("Connection error, please check the internet.")
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    dataPath = os.path.join(path, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    df = pd.read_csv(os.path.join(dataPath, "ptbxl_database.csv"), index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    data = [os.path.join(dataPath,f) for f in df.filename_lr]
    data = np.array(data)
    # data = [wfdb.rdsamp(os.path.join(dataPath,f)) for f in df.filename_lr]
    # data = np.array([signal for signal, meta in data])
    agg_df = pd.read_csv(os.path.join(dataPath,'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
    test_fold = 10
    source_train = data[np.where(df.strat_fold != test_fold)]
    target_train = df[(df.strat_fold != test_fold)].diagnostic_superclass
    source_test = data[np.where(df.strat_fold == test_fold)]
    target_test = df[(df.strat_fold == test_fold)].diagnostic_superclass
    trainset = {
        "source": source_train.tolist(),
        "target": target_train.to_numpy().tolist()
    }
    testset = {
        "source": source_test.tolist(),
        "target": target_test.to_numpy().tolist()
    }
    return trainset, testset