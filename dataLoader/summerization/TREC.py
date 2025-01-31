import os
import shutil
import warnings
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file, xml_to_dict


# tested by tjl 2025/1/30
def getTREC(path):
    urls = [
        'https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part1.zip',
        'https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part2.zip',
        'https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part3.zip',
        'https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part4.zip',
        'https://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.part5.zip',
    ]
    return datasetLoad(urls, path=path, datasetName="TREC")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            for url in urls:
                fn = url.split("/")[-1]
                print('downloading '+fn+'...')
                download_file(url, os.path.join(datasetPath, fn), datasetPath)
            
            for subdir in os.listdir(datasetPath):
                for f in os.listdir(os.path.join(datasetPath, subdir)):
                    shutil.move(os.path.join(datasetPath, subdir, f), os.path.join(datasetPath, f))
                shutil.rmtree(os.path.join(datasetPath, subdir))

            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    subpaths = [os.path.join(path,sub) for sub in os.listdir(path)]
    subpaths = [sub for sub in subpaths if os.path.isdir(sub)]

    dataset = []
    for subpath in subpaths:
        for fn in os.listdir(subpath):
            xml_path = os.path.join(subpath, fn)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            dataset.append(xml_to_dict(root))
    
    return dataset
