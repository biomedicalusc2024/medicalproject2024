import os
import re
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/19
def getROND(path):
    urls = ["https://raw.githubusercontent.com/Mayo-Clinic-RadOnc-Foundation-Models/Radiation-Oncology-NLP-Database/main/5-Question%20and%20answering%20(QA)/Medical-Physics-100questions-QA-format.csv"]
    return datasetLoad(urls=urls, path=path, datasetName="ROND")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(datasetPath, "ROND_QuestionAnswering.csv")
        if os.path.exists(datasetFile):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetFile)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], datasetFile)
            return loadLocalFiles(datasetFile)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df = pd.read_csv(path)
    df["Question"] = df["Question"].apply(lambda x: re.sub(r"^\d+\.\s*", "", x))
    df["Answer_choice"] = df["Answer_choice"].apply(lambda x: re.sub(r"^[A-Z]\.\s*", "", x))
    dataset = df.to_dict(orient='records')
    return dataset