import os
import shutil
import requests
import warnings
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getVQA_RAD(path):
    url = "https://files.osf.io/v1/resources/89kps/providers/osfstorage/?zip="
    return datasetLoad(url=url, path=path, datasetName="VQA-RAD")


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
    df = pd.read_json(os.path.join(path, "VQA_RAD Dataset Public.json"))
    source_cols = ['qid', 'phrase_type', 'qid_linked_id', 'image_case_url', 'image_name',
                   'image_organ', 'evaluation', 'question', 'question_rephrase',
                   'question_relation', 'question_frame', 'question_type']
    target_cols = ['answer', 'answer_type']
    dataset = {
        "source": df[source_cols].to_numpy().tolist(),
        "target": df[target_cols].to_numpy().tolist()
    }
    return dataset