import os
import zipfile
import subprocess
import warnings
from tqdm import tqdm
from collections import defaultdict

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

    # Ensure dataset contains all necessary categories
    if not dataset or not all(dataset.get(cat) for cat in ["benign", "malignant", "normal"]):
        raise ValueError("Failed to fetch or process the Breast Ultrasound Images Dataset.")

    return dataset["benign"], dataset["malignant"], dataset["normal"]


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

            return loadLocalFiles(datasetPath)

    except subprocess.CalledProcessError as e:
        print_sys(f"Error during Kaggle CLI execution: {e}")
        return None, None
    except Exception as e:
        print_sys(f"Error: {e}")
        return None, None


def loadLocalFiles(path):
    """
    Process the local files into a structured format based on the folder categories.
    """
    dataset = defaultdict(list)

    # Navigate to the nested folder
    nested_dir = os.path.join(path, "Dataset_BUSI_with_GT")
    if not os.path.isdir(nested_dir):
        raise ValueError(f"Expected folder 'Dataset_BUSI_with_GT' not found in {path}")

    # Iterate over the main categories (benign, malignant, normal)
    for category in ["benign", "malignant", "normal"]:
        category_path = os.path.join(nested_dir, category)
        if os.path.isdir(category_path):  # Check if the category folder exists
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)
                if file_name.endswith((".png", ".jpg", ".jpeg")):  # Assuming images are in these formats
                    # Add file path to the respective category
                    dataset[category].append(file_path)
        else:
            print_sys(f"Warning: No directory found for category '{category}'.")

    # Debugging outputs
    print_sys(f"Categories found: {list(dataset.keys())}")
    for key, files in dataset.items():
        print_sys(f"Category '{key}': {len(files)} files")

    return None, dataset
