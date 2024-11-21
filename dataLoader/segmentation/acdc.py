import os
import re
import requests
import zipfile
import warnings
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getACDC(path):
    """
    fetch ACDC data, process and return in following format:
        source (Pandas Series): a list of the segmentation sources
        source_idx (Pandas Series): a list of the segmentation sources index
        source_name (Pandas Series): a list of the segmentation sources names
        target (Pandas Series): a list of the segmentation target
    """
    url = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/637218e573e9f0047faa00fc/download"
    rawdata, dataset = datasetLoad(url=url, path=path, datasetName="ACDC")
    return rawdata, dataset['training'], dataset['testing']


def datasetLoad(url, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(datasetPath, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(os.path.join(datasetPath, "database"))
        else:
            print_sys("Downloading...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                os.makedirs(datasetPath, exist_ok=True)
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
                return loadLocalFiles(os.path.join(datasetPath, "database"))
            else:
                print_sys("Connection error, please check the internet.")
    except Exception as e:
        print_sys(f"error: {e}")
        
def loadLocalFiles(path):
    all_paths = defaultdict(dict)
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
                                dataset[item1].append([source_path, target_path])
                    idx = re.findall(r'\d+', item2)
                    idx = int(idx[0])
                    all_paths[item1][idx] = item_paths
    return all_paths, dataset