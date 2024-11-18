import os
import tarfile
import zipfile
import warnings
from tqdm import tqdm
from collections import defaultdict
import kagglehub

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getBraTS(path):
    """
    Fetch BraTS 2021 data, process, and return in the following format:
        source (Pandas Series): a list of the segmentation sources
        source_idx (Pandas Series): a list of the segmentation sources index
        source_name (Pandas Series): a list of the segmentation sources names
        target (Pandas Series): a list of the segmentation target
    """
    # URL to Kaggle dataset (requires Kaggle API setup)
    kaggle_url = "dschettler8845/brats-2021-task1"
    rawdata, dataset = datasetLoad(url=kaggle_url, path=path, datasetName="BraTS")
    return rawdata, dataset['training'], dataset['testing']


def datasetLoad(url, path, datasetName):
    """
    Download and process the dataset if not already available locally.
    """
    try:
        datasetPath = os.path.join(path, datasetName)

        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            print_sys("Downloading BraTS dataset...")
            os.makedirs(datasetPath, exist_ok=True)

            # Download the ZIP file using Kaggle API
            os.system(f"kaggle datasets download -d {url} -p {datasetPath}")

            # Extract the main ZIP file
            zip_path = os.path.join(datasetPath, "brats-2021-task1.zip")
            if os.path.exists(zip_path):
                print_sys("Extracting dataset ZIP...")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(datasetPath)
                print_sys("Extraction of ZIP complete.")
                os.remove(zip_path)

            # Extract .tar files
            for file_name in os.listdir(datasetPath):
                if file_name.endswith(".tar"):
                    tar_path = os.path.join(datasetPath, file_name)
                    print_sys(f"Extracting {file_name}...")
                    with tarfile.open(tar_path, "r") as tar:
                        tar.extractall(datasetPath)
                    print_sys(f"Extraction of {file_name} complete.")
                    os.remove(tar_path)

            return loadLocalFiles(datasetPath)

    except Exception as e:
        print_sys(f"Error: {e}")
        return None, None


def loadLocalFiles(path):
    """
    Process the local files into structured format.
    """
    all_paths = defaultdict(dict)
    dataset = defaultdict(list)

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith("_seg.nii") or file.endswith("_segmentation.nii"): 
                source_file = file.replace("_seg", "")
                source_path = os.path.join(root, source_file)
                target_path = os.path.join(root, file)

                if os.path.exists(source_path):
                    dataset["training"].append([source_path, target_path])

    return all_paths, dataset
