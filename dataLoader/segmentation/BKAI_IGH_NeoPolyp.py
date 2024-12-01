import os
import zipfile
import subprocess
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def getBKAI_IGH(path):
    """
    Fetch BKAI-IGH NeoPolyp Dataset, process, and return in the following format:
        rawdata (list): A list of all extracted file paths.
    """
    kaggle_competition = "bkai-igh-neopolyp"
    rawdata = datasetLoad(competition=kaggle_competition, path=path, datasetName="BKAI_IGH")

    if not rawdata:
        raise ValueError("Failed to fetch or process the BKAI-IGH NeoPolyp Dataset.")

    return rawdata


def datasetLoad(competition, path, datasetName):
    """
    Download and process the dataset if not already available locally.

    Args:
        competition (str): Kaggle competition name.
        path (str): Base directory to store the dataset.
        datasetName (str): Name of the dataset.

    Returns:
        rawdata (list): A list of all extracted file paths.
    """
    datasetPath = os.path.join(path, datasetName)
    rawdata = []

    try:
        if os.path.exists(datasetPath):
            print("Found local copy...")
            rawdata = loadLocalFiles(datasetPath)
        else:
            print("Downloading BKAI-IGH NeoPolyp Dataset...")
            os.makedirs(datasetPath, exist_ok=True)

            # Use Kaggle CLI to download the dataset
            kaggle_command = f"kaggle competitions download -c {competition} -p {datasetPath}"
            subprocess.run(kaggle_command, shell=True, check=True)

            # Identify the ZIP files in the datasetPath
            zip_files = [f for f in os.listdir(datasetPath) if f.endswith(".zip")]
            if not zip_files:
                raise FileNotFoundError("No ZIP file found after download.")

            # Extract all ZIP files found
            for zip_file in zip_files:
                zip_path = os.path.join(datasetPath, zip_file)
                print(f"Extracting dataset from {zip_path}...")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(datasetPath)
                os.remove(zip_path)
            print("Extraction complete.")

            rawdata = loadLocalFiles(datasetPath)

    except subprocess.CalledProcessError as e:
        print(f"Error during Kaggle CLI execution: {e}")
    except Exception as e:
        print(f"Error: {e}")

    return rawdata


def loadLocalFiles(path):
    """
    Process the local files into a flat list of file paths.

    Args:
        path (str): Path to the directory containing files.

    Returns:
        list: A list of all file paths.
    """
    rawdata = []

    # Traverse all files in the directory
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            rawdata.append(file_path)

    print(f"Total files collected: {len(rawdata)}")
    return rawdata
