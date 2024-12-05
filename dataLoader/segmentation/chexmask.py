import os
import zipfile
import warnings
from tqdm import tqdm
import requests
from collections import defaultdict

warnings.filterwarnings("ignore")

from ..utils import print_sys


def getCheXmask(path):
    """
    Fetch CheXmask Dataset, process, and return in the following format:
        rawdata (list): A list of all extracted file paths.
    """
    urls = {
        "cheXmask file": "https://physionet.org/static/published-projects/chexmask-cxr-segmentation-data/chexmask-database-a-large-scale-dataset-of-anatomical-segmentation-masks-for-chest-x-ray-images-0.1.zip",
    }

    rawdata = datasetLoad(urls=urls, path=path, datasetName="CheXmask")

    if not rawdata:
        raise ValueError("Failed to fetch or process the CheXmask Dataset.")

    return rawdata


def datasetLoad(urls, path, datasetName):
    """
    Download and process the dataset if not already available locally.

    Args:
        urls (dict): Dictionary of dataset names and their URLs.
        path (str): Base directory to save the files.
        datasetName (str): Name of the dataset.

    Returns:
        rawdata (list): List of all extracted file paths.
    """
    datasetPath = os.path.join(path, datasetName)
    rawdata = []

    os.makedirs(datasetPath, exist_ok=True)

    for task, url in urls.items():
        file_name = f"{task}.zip"
        file_path = os.path.join(datasetPath, file_name)

        # Download the file if not already present
        if not os.path.exists(file_path):
            print(f"Downloading {task}...")
            download_file(url, file_path)

        # Extract the file if it hasn't been extracted yet
        extracted_dir = os.path.join(datasetPath, task)
        if not os.path.exists(extracted_dir):
            print(f"Extracting {file_name}...")
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(path=extracted_dir)
                print(f"Extraction complete for {file_name}.")
            except zipfile.BadZipFile as e:
                print(f"Failed to extract {file_name}. Error: {e}")
                continue

        # Collect all extracted file paths
        for root, _, files in os.walk(extracted_dir):
            for file in files:
                rawdata.append(os.path.join(root, file))

    print(f"Total files collected: {len(rawdata)}")
    return rawdata


def download_file(url, file_path):
    """
    Download a file from the given URL and save it to the specified path.

    Args:
        url (str): URL of the file to download.
        file_path (str): Path to save the downloaded file.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status() 

    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, "wb") as file, tqdm(
        total=total_size, unit="iB", unit_scale=True, desc=file_path.split("/")[-1]
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))


def loadLocalFiles(path):
    """
    Process the local files into a flat list of file paths.
    """
    rawdata = []

    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            rawdata.append(file_path)

    print_sys(f"Total files collected: {len(rawdata)}")
    return rawdata
