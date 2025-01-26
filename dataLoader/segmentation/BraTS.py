import os
import tarfile
import warnings

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/22
def getBraTS(path):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/dschettler8845/brats-2021-task1"]
    return datasetLoad(urls=urls, path=path, datasetName="BraTS")


def datasetLoad(urls, path, datasetName):
    """
    Download and process the dataset if not already available locally.
    """
    try:
        datasetPath = os.path.join(path, datasetName)
        zip_path = os.path.join(datasetPath, "brats-2021-task1.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], zip_path, datasetPath)

            # Extract .tar files
            for file_name in os.listdir(datasetPath):
                if "Training" in file_name:
                    tar_path = os.path.join(datasetPath, file_name)
                    print_sys(f"Extracting {file_name}...")
                    with tarfile.open(tar_path, "r") as tar:
                        tar.extractall(datasetPath)
                    print_sys(f"Extraction of {file_name} complete.")
                    os.remove(tar_path)
                else:
                    extractPath = file_name.split(".")[0]
                    extractPath = os.path.join(datasetPath, extractPath)
                    tar_path = os.path.join(datasetPath, file_name)
                    print_sys(f"Extracting {file_name}...")
                    with tarfile.open(tar_path, "r") as tar:
                        tar.extractall(extractPath)
                    print_sys(f"Extraction of {file_name} complete.")
                    os.remove(tar_path)

            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"Error: {e}")


def loadLocalFiles(path):
    dataset = []

    subpaths = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path,p))]
    for subpath in subpaths:
        subpathfull = os.path.join(path, subpath)
        ele_dict = {}
        for fn in os.listdir(subpathfull):
            col = (fn.split("_")[-1]).split(".")[0]
            ele_dict[col] = os.path.join(subpathfull, fn)
        dataset.append(ele_dict)

    return dataset
