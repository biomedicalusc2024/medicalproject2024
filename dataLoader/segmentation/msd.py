import os
import tarfile
import warnings
from tqdm import tqdm
from collections import defaultdict
import gdown

warnings.filterwarnings("ignore")

from ..utils import print_sys


def getMSD(path):
    """
    Fetch MSD (Medical Segmentation Decathlon) Dataset, process, and return in the following format:
        rawdata (list): A list of all extracted file paths.
    """
    urls = {
        "Task01_BrainTumour": "https://drive.google.com/uc?id=1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU",
        "Task02_Heart": "https://drive.google.com/uc?id=1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY",
        "Task03_Liver": "https://drive.google.com/uc?id=1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu",
        "Task04_Hippocampus": "https://drive.google.com/uc?id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C",
        "Task05_Prostate": "https://drive.google.com/uc?id=1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a",
        "Task06_Lung": "https://drive.google.com/uc?id=1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi",
        "Task07_Pancreas": "https://drive.google.com/uc?id=1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL",
        "Task08_HepaticVessel": "https://drive.google.com/uc?id=1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS",
        "Task09_Spleen": "https://drive.google.com/uc?id=1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE",
        "Task10_Colon": "https://drive.google.com/uc?id=1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y",
    }

    rawdata = datasetLoad(urls=urls, path=path, datasetName="MSD")

    if not rawdata:
        raise ValueError("Failed to fetch or process the MSD Dataset.")

    return rawdata


def datasetLoad(urls, path, datasetName):
    """
    Download and process the dataset if not already available locally.

    Args:
        urls (dict): Dictionary of dataset names and their Google Drive URLs.
        path (str): Base directory to store downloaded files.
        datasetName (str): Name of the dataset.

    Returns:
        rawdata (list): List of all extracted file paths.
    """
    datasetPath = os.path.join(path, datasetName)
    rawdata = []

    os.makedirs(datasetPath, exist_ok=True)

    for task, url in urls.items():
        file_name = f"{task}.tar"
        file_path = os.path.join(datasetPath, file_name)

        # Download the file if not already present
        if not os.path.exists(file_path):
            print(f"Downloading {task}...")
            gdown.download(url, file_path, quiet=False)

        # Extract the file if it hasn't been extracted yet
        extracted_dir = os.path.join(datasetPath, task)
        if not os.path.exists(extracted_dir):
            print(f"Extracting {file_name}...")
            try:
                with tarfile.open(file_path, "r") as tar:
                    tar.extractall(path=extracted_dir)
                print(f"Extraction complete for {file_name}.")
            except tarfile.ReadError as e:
                print(f"Failed to extract {file_name}. Error: {e}")
                continue

        # Collect all extracted file paths
        for root, _, files in os.walk(extracted_dir):
            for file in files:
                rawdata.append(os.path.join(root, file))

    print(f"Total files collected: {len(rawdata)}")
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
