import os
import zipfile
import subprocess
import warnings
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings("ignore")

from ..utils import print_sys


def getChestXray(path):
    """
    Fetch Pancreas CT Dataset, process, and return in the following format:
        rawdata (list): A list of all extracted file paths.
    """
    kaggle_url = "nikhilpandey360/chest-xray-masks-and-labels"
    rawdata = datasetLoad(url=kaggle_url, path=path, datasetName="chestXray")

    if not rawdata:
        raise ValueError("Failed to fetch or process the chest X-rays Dataset.")

    return rawdata


def datasetLoad(url, path, datasetName):
    """
    Download and process the dataset if not already available locally.
    """
    datasetPath = os.path.join(path, datasetName)
    rawdata = []

    try:
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            rawdata = loadLocalFiles(datasetPath)
        else:
            print_sys("Downloading chest X-rays Dataset...")
            os.makedirs(datasetPath, exist_ok=True)

            # Use Kaggle CLI to download the dataset
            kaggle_command = f"kaggle datasets download -d {url} -p {datasetPath}"
            subprocess.run(kaggle_command, shell=True, check=True)

            # Identify the ZIP file in the datasetPath
            zip_files = [f for f in os.listdir(datasetPath) if f.endswith(".zip")]
            if not zip_files:
                raise FileNotFoundError("No ZIP file found after download.")

            # Extract the first ZIP file found
            zip_path = os.path.join(datasetPath, zip_files[0])
            print_sys(f"Extracting dataset from {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(datasetPath)
            print_sys("Extraction complete.")

            rawdata = loadLocalFiles(datasetPath)

    except subprocess.CalledProcessError as e:
        print_sys(f"Error during Kaggle CLI execution: {e}")
    except Exception as e:
        print_sys(f"Error: {e}")

    return rawdata


def loadLocalFiles(path):
    """
    Process the local files into a flat list of file paths.
    """
    rawdata = []

    # Traverse all files in the directory
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            rawdata.append(file_path)

    print_sys(f"Total files collected: {len(rawdata)}")
    return rawdata
