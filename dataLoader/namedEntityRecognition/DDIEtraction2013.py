import os
import warnings
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

# TO DO: not clear how to use data downloaded, need further explanation

def getDDIEtraction2013(path):
    urls = ["https://github.com/isegura/DDICorpus/raw/master/DDICorpus-2013.zip"]
    return datasetLoad(urls=urls, path=path, datasetName="DDIEtraction2013")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "DDICorpus-2013.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            download_file(urls[0], datasetZip, datasetPath)
            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    dataPath = os.path.join(path, "DDICorpus")
    trainDataPath = os.path.join(dataPath, "Train")
    testDataPath = os.path.join(dataPath, "Test")
    breakpoint()
    