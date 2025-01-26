import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/25
def getCrossDocked2020(path):
    # urls = [
    #     "http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3.tgz",
    #     "http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3_types.tgz",
    # ]

    urls = [
        "http://bits.csb.pitt.edu/files/crossdock2020/downsampled_CrossDocked2020_v1.3.tgz",
        "http://bits.csb.pitt.edu/files/crossdock2020/downsampled_CrossDocked2020_v1.3_types.tgz",
    ]
    
    return datasetLoad(urls, path=path, datasetName="CrossDocked2020")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        dockerPath = os.path.join(datasetPath, "dockers")
        typesPath = os.path.join(datasetPath, "types")
        if (os.path.exists(dockerPath) and os.path.exists(typesPath)):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath)
            
            fn = urls[0].split("/")[-1]
            download_file(urls[0], os.path.join(datasetPath, fn), os.path.join(datasetPath, "dockers"))

            fn = urls[1].split("/")[-1]
            download_file(urls[1], os.path.join(datasetPath, fn), os.path.join(datasetPath))

            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    dockersPath, typesPath = os.path.join(path, "dockers"), os.path.join(path, "types")
    # splits = ["test0", "test1", "test2", "train0", "train1", "train2"]
    # typesNames = []
    # typesNames += [f"cdonly_it2_tt_v1.3_0_{x}.types" for x in splits]
    # typesNames += [f"it0_tt_v1.3_0_{x}.types" for x in splits]
    # typesNames += [f"it2_redocked_tt_v1.3_0_{x}.types" for x in splits]
    # typesNames += [f"it2_redocked_tt_v1.3_completeset_{x}.types" for x in ["test0", "train0"]]
    # typesNames += [f"it2_tt_v1.3_0_{x}.types" for x in splits]
    # typesNames += [f"it2_tt_v1.3_10p20n_{x}.types" for x in splits]
    # typesNames += [f"it2_tt_v1.3_completeset_{x}.types" for x in ["test0", "train0"]]
    # typesNames += [f"mod_it2_tt_v1.3_0_{x}.types" for x in splits]
    # typesNames += [f"mod_it2_tt_v1.3_completeset_{x}.types" for x in ["test0", "train0"]]

    splits = ["test0", "test1", "test2", "train0", "train1", "train2"]
    names = ["label", "pK", "RMSD_to_csystal", "Receptor", "Ligand", "Autodock_Vina_score"]
    typesNames = {}
    typesNames.update({x:os.path.join(typesPath, f"it2_tt_v1.3_10p20n_{x}.types") for x in splits})

    dfs = {sub:pd.read_csv(typesNames[sub], header=None, names=names, sep=" ") for sub in splits}
    df_train = pd.concat([dfs["train0"], dfs["train1"], dfs["train2"]], ignore_index=True)
    df_test = pd.concat([dfs["test0"], dfs["test1"], dfs["test2"]], ignore_index=True)

    df_train["Receptor"] = df_train["Receptor"].apply(lambda x: os.path.join(dockersPath,x))
    df_train["Ligand"] = df_train["Ligand"].apply(lambda x: os.path.join(dockersPath,x))
    # df_train = df_train[df_train["Receptor"].apply(lambda x: os.path.exists(x))]
    # df_train = df_train[df_train["Ligand"].apply(lambda x: os.path.exists(x))]
    df_test["Receptor"] = df_test["Receptor"].apply(lambda x: os.path.join(dockersPath,x))
    df_test["Ligand"] = df_test["Ligand"].apply(lambda x: os.path.join(dockersPath,x))
    # df_test = df_test[df_test["Receptor"].apply(lambda x: os.path.exists(x))]
    # df_test = df_test[df_test["Ligand"].apply(lambda x: os.path.exists(x))]
    
    trainset = df_train.to_dict(orient="records")
    testset = df_test.to_dict(orient="records")

    return trainset, testset
    