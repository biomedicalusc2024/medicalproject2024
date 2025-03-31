import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

# Dataset index for NHIS, organized by year and category 
NHIS_INDEX = {
    "2023": {
        "SampleAdult": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip",
        "SampleChild": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/child23csv.zip",
        "ImputedIncomeAdult": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adultinc23csv.zip",
        "ImputedIncomeChild": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/childinc23csv.zip",
        "ParaData": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/paradata23csv.zip"
    },
    "2022": {
        "SampleAdult": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2022/adult22csv.zip",
        "SampleChild": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2022/child22csv.zip",
        "ImputedIncomeAdult": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2022/adultinc22csv.zip",
        "ImputedIncomeChild": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2022/childinc22csv.zip",
        "ParaData": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2022/paradata22csv.zip"
    }
}

# Main loader function
def getNHIS(path, year="2023", category="SampleAdult"):
    dataset_url = NHIS_INDEX.get(year, {}).get(category, None)
    if not dataset_url:
        print_sys(f"Dataset not found for year={year}, category={category}")
        return None

    datasetPath = os.path.join(path, "NHIS", year, category)
    os.makedirs(datasetPath, exist_ok=True)

    file_name = dataset_url.split("/")[-1]
    file_path = os.path.join(datasetPath, file_name)

    if not os.path.exists(file_path):
        print_sys(f"Downloading NHIS {category} data for {year}")
        download_file(dataset_url, file_path, datasetPath)
    else:
        print_sys(f"Found local file: {file_name}")

    return loadLocalFile(file_path)

# File loader

def loadLocalFile(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".csv":
            df = pd.read_csv(file_path)

        elif ext == ".tsv":
            df = pd.read_csv(file_path, sep="\t")

        elif ext in [".dat", ".txt"]:
            print_sys(f"Attempting to load fixed-width file: {file_path}")

            df = pd.read_csv(file_path, sep=",", engine="python", error_bad_lines=False)

        elif ext == ".zip":
            print_sys("Zip file detected. Please unzip and retry loading manually.")
            return None

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        df["__source_file__"] = os.path.basename(file_path)
        return df.to_dict(orient="records")

    except Exception as e:
        print_sys(f"Error loading file: {e}")
        return None


# Helper to list available datasets
def list_nhis_datasets():
    print("\U0001F4CB Available NHIS datasets:")
    for year, categories in NHIS_INDEX.items():
        print(f"  {year}:")
        for category in categories:
            print(f"    - {category}")
