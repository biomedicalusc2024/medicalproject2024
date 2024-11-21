import os
import re
import requests
import tarfile
import warnings
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings("ignore")

from ..utils import print_sys


def getCIR(path):
    """
    Fetch CIR data, process, and return in the following format:
        rawdata (list): A list of all extracted file paths.
    """
    urls = {
        "CIRDataset_LCSR": "https://zenodo.org/records/6762573/files/CIRDataset_LCSR.tar.bz2?download=1",
        "CIRDataset_npy_for_cnn": "https://zenodo.org/records/6762573/files/CIRDataset_npy_for_cnn.tar.bz2?download=1",
        "CIRDataset_pickle_for_voxel2mesh": "https://zenodo.org//records/6762573/files/CIRDataset_pickle_for_voxel2mesh.tar.bz2?download=1",
        "pretrained_model_mesh_encoder": "https://zenodo.org//records/6762573/files/pretrained_model-mesh+encoder.tar.bz2?download=1",
        "pretrained_model_meshonly": "https://zenodo.org//records/6762573/files/pretrained_model-meshonly.tar.bz2?download=1"
    }

    return datasetLoad(urls=urls, path=path, datasetName="CIR")

def datasetLoad(urls, path, datasetName):
    """
    Downloads and extracts CIR dataset from multiple URLs.

    Args:
        urls (dict): Dictionary with dataset parts as keys and their URLs as values.
        path (str): Base directory to store downloaded files.
        datasetName (str): Name of the dataset.

    Returns:
        rawdata (list): List of all extracted file paths.
    """
    datasetPath = os.path.join(path, datasetName)
    rawdata = []

    os.makedirs(datasetPath, exist_ok=True)

    for key, url in urls.items():
        file_name = url.split("/")[-1]
        file_path = os.path.join(datasetPath, file_name)

        # Download the file if not already present
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            with open(file_path, "wb") as f, tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Extract the file if it hasn't been extracted yet
        extracted_dir = os.path.join(datasetPath, key)
        if not os.path.exists(extracted_dir):
            print(f"Extracting {file_name}...")
            try:
                with tarfile.open(file_path, "r:bz2") as tar:
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



def downloadFile(url, file_path):
    """
    Download a file from the given URL and save it to the specified path.
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(file_path, "wb") as file:
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
            print_sys(f"Downloaded {file_path}")
        else:
            raise ValueError(f"Failed to download {url}: {response.status_code}")
    except Exception as e:
        print_sys(f"Error downloading file from {url}: {e}")


def extractTarFile(tar_file, extract_path):
    """
    Extract a .tar.bz2 file to the specified directory.
    """
    try:
        with tarfile.open(tar_file, "r:bz2") as tar:
            tar.extractall(path=extract_path)
        print_sys(f"Extracted {tar_file}")
        os.remove(tar_file)
    except Exception as e:
        print_sys(f"Error extracting {tar_file}: {e}")


def loadLocalFiles(path):
    """
    Process the local files into a structured format based on the folder structure.
    """
    dataset = defaultdict(list)

    # Iterate over the dataset files
    for file_name in os.listdir(path):
        if file_name.endswith(".tar.bz2"):
            continue  # Skip remaining .tar.bz2 files if any exist
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            dataset[file_name] = []
            for root, _, files in os.walk(file_path):
                for file in files:
                    dataset[file_name].append(os.path.join(root, file))
        else:
            print_sys(f"Warning: {file_name} is not a directory and is skipped.")

    # Debugging outputs
    print_sys(f"Categories found: {list(dataset.keys())}")
    for key, files in dataset.items():
        print_sys(f"Category '{key}': {len(files)} files")

    return None, dataset
