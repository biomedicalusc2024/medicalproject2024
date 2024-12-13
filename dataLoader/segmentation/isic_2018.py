import os
import requests
import zipfile
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def getISIC_2018(path):
    """
    Fetch ISIC-2018 data from multiple URLs, process, and return in the following format:
        rawdata (list): A list of all extracted file paths.
    """
    urls = {
        "ISIC2018_Part1": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
        "ISIC2018_Part2": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip",
        "ISIC2018_Part3": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_LesionGroupings.csv",
    }

    return datasetLoad(urls=urls, path=path, datasetName="ISIC-2018")


def datasetLoad(urls, path, datasetName):
    """
    Download and process the ISIC dataset from multiple URLs.

    Args:
        urls (dict): Dictionary containing dataset names and their URLs.
        path (str): Base directory to store the downloaded files.
        datasetName (str): Name of the dataset.

    Returns:
        rawdata (list): List of all processed file paths.
    """
    datasetPath = os.path.join(path, datasetName)
    rawdata = []

    os.makedirs(datasetPath, exist_ok=True)

    for name, url in urls.items():
        file_extension = "zip" if url.endswith(".zip") else "csv"
        file_name = f"{name}.{file_extension}"
        file_path = os.path.join(datasetPath, file_name)

        # Download the file if not already present
        if not os.path.exists(file_path):
            print(f"Starting download: {file_name}")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                with open(file_path, "wb") as f, tqdm(
                    total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {file_name}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                print(f"Download complete: {file_name}")
            else:
                print(f"Failed to download {file_name}. HTTP Status Code: {response.status_code}")
                continue

        # Handle .zip files by extracting them
        if file_extension == "zip":
            extracted_dir = os.path.join(datasetPath, name)
            if not os.path.exists(extracted_dir):
                print(f"Starting extraction: {file_name}")
                try:
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(path=extracted_dir)
                    print(f"Extraction complete: {file_name}")
                except zipfile.BadZipFile as e:
                    print(f"Failed to extract {file_name}. Error: {e}")
                    continue

            for root, _, files in os.walk(extracted_dir):
                for file in files:
                    rawdata.append(os.path.join(root, file))

        # Handle .csv files directly
        elif file_extension == "csv":
            print(f"Processing CSV file: {file_name}")
            rawdata.append(file_path)

    print(f"Total files collected: {len(rawdata)}")
    return rawdata
