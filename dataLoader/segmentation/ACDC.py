import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/22
def getACDC(path):
    urls = ["https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/637218e573e9f0047faa00fc/download"]
    return datasetLoad(urls=urls, path=path, datasetName="ACDC")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(datasetPath, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(os.path.join(datasetPath, "database"))
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], datasetZip, datasetPath)
            return loadLocalFiles(os.path.join(datasetPath, "database"))
    except Exception as e:
        print_sys(f"error: {e}")
        

def loadLocalFiles(path):
    dataset = defaultdict(list)
    for item1 in os.listdir(path):
        if item1 in ["training", "testing"]:
            full_path1 = os.path.join(path, item1)
            for item2 in os.listdir(full_path1):
                full_path2 = os.path.join(full_path1, item2)
                if os.path.isdir(full_path2):
                    item_paths = []
                    path_list3 = os.listdir(full_path2)
                    for item3 in path_list3:
                        full_path3 = os.path.join(full_path2, item3)
                        if ".md" not in item3:
                            item_paths.append(full_path3)
                            if "_gt" in item3:
                                source_speci = item3.split("_gt")[0]
                                source_path = [item for item in path_list3 if (source_speci in item and "_gt" not in item)][0]
                                source_path = os.path.join(full_path2, source_path)
                                target_path = os.path.join(full_path2, item3)
                                dataset[item1].append({"source_path":source_path, "target_path":target_path})

    return dataset["training"], dataset["testing"]