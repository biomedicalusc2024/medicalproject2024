import os
import shutil
import warnings
import pandas as pd
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file, xml_to_dict


# tested by tjl 2025/1/30
def getLiveQA_TREC_2017(path):
    urls = ["https://github.com/abachaa/LiveQA_MedicalTask_TREC2017/archive/refs/heads/master.zip"]
    return datasetLoad(urls=urls, path=path, datasetName="LiveQA_TREC_2017")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            download_file(urls[0], datasetZip, datasetPath)
            tempdir = os.path.join(datasetPath, "LiveQA_MedicalTask_TREC2017-master", "TestDataset")
            for filename in os.listdir(tempdir):
                shutil.move(os.path.join(tempdir, filename), datasetPath)
            shutil.rmtree(os.path.join(datasetPath, "LiveQA_MedicalTask_TREC2017-master"))

            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    file_path = os.path.join(path, "TREC-2017-LiveQA-Medical-Test.xml")
    tree = ET.parse(file_path)
    root = tree.getroot()
    dataset = xml_to_dict(root)
    return dataset['NLM-QUESTION']

