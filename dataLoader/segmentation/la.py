import os
import requests
import zipfile
import warnings
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings("ignore")

from ..utils import print_sys


def getLA(path):
    """
    Fetch LA data, process, and return in the following format:
        rawdata (list): A list of all extracted file paths.
    """
    url = "https://ucc27133c5fe1689324a65c3a974.dl-au.dropboxusercontent.com/cd/0/get/CfL9DAx0HMdgdvx5dv-CiYKSjCGEYxqL5Fo_RTF4QJTPF6r9Qjem-LDbyp5MY4D72LcfjOca-5QECsRSm2Ujy-wpHRhWE44ErScoFgI47IlEztS8Snihm1tL6nmtjtRMvl_uzO9YfgPUYm07r3LkqBR250B7fD_jtV6YDKElgdrjyw/file?_download_id=485465245481997439970136929159236780617455625347326127184307358&_log_download_success=1&_notify_domain=www.dropbox.com&dl=1"  # Replace with the actual URL

    return datasetLoad(url=url, path=path, datasetName="LA")


def datasetLoad(url, path, datasetName):
    """
    Downloads and extracts the LA dataset from a single URL.

    Args:
        url (str): URL of the dataset zip file.
        path (str): Base directory to store the downloaded file.
        datasetName (str): Name of the dataset.

    Returns:
        rawdata (list): List of all extracted file paths.
    """
    datasetPath = os.path.join(path, datasetName)
    rawdata = []

    os.makedirs(datasetPath, exist_ok=True)

    file_name = os.path.basename(url).split("?")[0]  # Extract clean file name
    file_path = os.path.join(datasetPath, file_name)

    # Download the file if not already present
    if not os.path.exists(file_path):
        print(f"Starting download: {file_name}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            print(f"File size: {total_size / 1024:.2f} KB")

            with open(file_path, "wb") as f, tqdm(
                total=total_size, unit="iB", unit_scale=True, desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            print(f"Download complete: {file_name}")
        else:
            print(f"Failed to download {file_name}. HTTP Status Code: {response.status_code}")
            return rawdata

    # Extract the file if it hasn't been extracted yet
    extracted_dir = os.path.join(datasetPath, "extracted")
    if not os.path.exists(extracted_dir):
        print(f"Starting extraction: {file_name}")
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(path=extracted_dir)
            print(f"Extraction complete: {file_name}")
        except zipfile.BadZipFile as e:
            print(f"Failed to extract {file_name}. Error: {e}")
            return rawdata

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
            total_size = int(response.headers.get("content-length", 0))
            print(f"Starting download: {file_path} (Size: {total_size / 1024:.2f} KB)")
            with open(file_path, "wb") as file:
                with tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            print_sys(f"Download completed for {file_path}")
        else:
            print_sys(f"Failed to download {url}: HTTP {response.status_code}")
    except Exception as e:
        print_sys(f"Error downloading file from {url}: {e}")


def extractZipFile(zip_file, extract_path):
    """
    Extract a .zip file to the specified directory.
    """
    try:
        print(f"Starting extraction: {zip_file}")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(path=extract_path)
        print(f"Extraction complete: {zip_file}")
        os.remove(zip_file)
    except Exception as e:
        print_sys(f"Error extracting {zip_file}: {e}")


def loadLocalFiles(path):
    """
    Process the local files into a structured format based on the folder structure.
    """
    dataset = defaultdict(list)

    # Iterate over the dataset files
    for file_name in os.listdir(path):
        if file_name.endswith(".zip"):
            continue  # Skip remaining .zip files if any exist
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