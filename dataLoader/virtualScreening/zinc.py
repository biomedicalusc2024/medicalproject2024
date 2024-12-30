import os
import shutil
import zipfile
import datasets
import requests
import pandas as pd
from tqdm import tqdm

from ..utils import print_sys

def getZINC(path):
    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    return datasetLoad(url=url, path=path, datasetName="ZINC")

def datasetLoad(url, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(datasetPath, "ZINC.csv")
        if os.path.exists(datasetFile):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetFile)
        else:
            print_sys("Downloading...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                os.makedirs(datasetPath, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(datasetFile, "wb") as file:
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
                return loadLocalFiles(datasetFile)
            else:
                print_sys("Connection error, please check the internet.")
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df = pd.read_csv(path)
    df["smiles"] = df["smiles"].apply(lambda x: x.strip('\n'))
    return {
        "source": df["smiles"].to_list(),
        "target": df[["logP", "qed", "SAS"]].to_numpy().tolist(),
    }