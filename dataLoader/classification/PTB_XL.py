import os
import ast
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/22
def getPTB_XL(path):
    urls = ["https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"]
    return datasetLoad(urls=urls, path=path, datasetName="PTB_XL")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            download_file(urls[0], datasetZip, datasetPath)
            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    dataPath = os.path.join(path, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    df = pd.read_csv(os.path.join(dataPath, "ptbxl_database.csv"), index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
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
    df = df.reset_index(drop=True)
    df["filename_lr"] = df["filename_lr"].apply(lambda x: os.path.join(dataPath,x))
    df_train = df[(df.strat_fold != test_fold)]
    df_test = df[(df.strat_fold == test_fold)]
    trainset = df_train.to_dict(orient='records')
    testset = df_test.to_dict(orient='records')
    return trainset, testset