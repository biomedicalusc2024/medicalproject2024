import os
import zipfile
import warnings
from tqdm import tqdm
from collections import defaultdict
import kagglehub

warnings.filterwarnings("ignore")

from ..utils import print_sys


def getBUID(path):
    """
    Fetch Breast Ultrasound Images Dataset, process, and return in the following format:
        benign (list): a list of image file paths for benign cases
        malignant (list): a list of image file paths for malignant cases
        normal (list): a list of image file paths for normal cases
    """
    kaggle_url = "aryashah2k/breast-ultrasound-images-dataset"
    rawdata, dataset = datasetLoad(url=kaggle_url, path=path, datasetName="BUID")
    if rawdata is None or dataset is None:
        raise ValueError("Failed to fetch or process the Breast Ultrasound Images Dataset.")
    return dataset['benign'], dataset['malignant'], dataset['normal']


def datasetLoad(url, path, datasetName):
    """
    Download and process the dataset if not already available locally.
    """
    datasetPath = os.path.join(path, datasetName)

    try:
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            print_sys("Downloading Breast Ultrasound Images Dataset...")
            os.makedirs(datasetPath, exist_ok=True)

            # Download the dataset using KaggleHub
            kagglehub.dataset_download(url, datasetPath)

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

            return loadLocalFiles(datasetPath)

    except Exception as e:
        print_sys(f"Error: {e}")
        return None, None


def loadLocalFiles(path):
    """
    Process the local files into a structured format based on the folder categories.
    """
    all_paths = defaultdict(dict)
    dataset = defaultdict(list)

    # Iterate over the main categories (benign, malignant, normal)
    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        if os.path.isdir(category_path):
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)
                if file_name.endswith((".png", ".jpg", ".jpeg")):  # Assuming images are in these formats
                    # In this case, no masks are present, so only the source path is added
                    dataset[category].append(file_path)

    # Flatten dataset structure for train/test compatibility
    all_paths = {
        "all": {i: dataset[category] for i, category in enumerate(dataset.keys())}
    }

    return all_paths, dataset
