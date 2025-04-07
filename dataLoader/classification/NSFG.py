import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

# NSFG dataset index 
NSFG_YEAR = "2022-2023"

NSFG_INDEX = {
    "FemaleRespondent": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/NSFG/NSFG-2022-2023-FemRespPUFData.zip"
    ],
    "Pregnancy": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/NSFG/NSFG-2022-2023-FemPregPUFData.zip"
    ],
    "MaleRespondent": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/NSFG/NSFG-2022-2023-MaleRespPUFData.zip"
    ]
}

def getNSFG(path):
    all_data = {}
    year_path = os.path.join(path, "NSFG", NSFG_YEAR.replace("-", "_"))

    for category, urls in NSFG_INDEX.items():
        category_data = []

        for dataset_url in urls:
            datasetPath = os.path.join(year_path, category)
            os.makedirs(datasetPath, exist_ok=True)

            file_name = dataset_url.split("/")[-1]
            file_path = os.path.join(datasetPath, file_name)

            if not os.path.exists(file_path):
                print_sys(f"Downloading NSFG file: {file_name}")
                download_file(dataset_url, file_path, datasetPath)
            else:
                print_sys(f"Found local file: {file_name}")

            records = loadLocalFile(file_path)
            if records:
                category_data.extend(records)

        all_data[category] = category_data

    return all_data

def loadLocalFile(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".tsv":
            df = pd.read_csv(file_path, sep="\t")
        elif ext in [".xpt"]:
            df = pd.read_sas(file_path, format="xport", encoding="utf-8")
        elif ext == ".zip":
            print_sys(f"Zip file detected: {file_path} — please unzip manually if needed.")
            return None
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        df["__source_file__"] = os.path.basename(file_path)
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        return train_df.to_dict(orient="records")

    except Exception as e:
        print_sys(f"Error loading file: {e}")
        return None
