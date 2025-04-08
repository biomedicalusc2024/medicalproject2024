import os
import warnings
import pandas as pd
import zipfile

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

# NEHRS dataset info
NEHRS_YEAR = "2021"

NEHRS_INDEX = {
    "Data": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NEHRS/nehrs2021-SAS.zip",
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NEHRS/nehrs2021-Stata.zip",
    ],
    "Documentation": [
        "hhttps://www.cdc.gov/nchs/data/nehrs/NEHRS2021Doc-508.pdf",
        "https://www.cdc.gov/nchs/data/nehrs/2021-NEHRS-public-use-file-layout-508.pdf",
        "https://www.cdc.gov/nchs/data/nehrs/NEHRS2020-Questionnaire-508.pdf",
    ]
}

def getNEHRS(path):
    all_data = {}
    year_path = os.path.join(path, "NEHRS", NEHRS_YEAR)

    for category, urls in NEHRS_INDEX.items():
        category_data = []

        for dataset_url in urls:
            datasetPath = os.path.join(year_path, category)
            os.makedirs(datasetPath, exist_ok=True)

            file_name = dataset_url.split("/")[-1]
            file_path = os.path.join(datasetPath, file_name)

            if not os.path.exists(file_path):
                print_sys(f"Downloading NEHRS file: {file_name}")
                download_file(dataset_url, file_path, datasetPath)
            else:
                print_sys(f"Found local file: {file_name}")

            # If data zip, load it
            if file_name.endswith(".zip") and category == "Data":
                records = loadLocalFile(file_path)
                if records:
                    category_data.extend(records)

        if category_data:
            all_data[category] = category_data

    return all_data

def loadLocalFile(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".zip":
            with zipfile.ZipFile(file_path, 'r') as z:
                zip_members = z.namelist()
                csv_files = [f for f in zip_members if f.lower().endswith(".csv")]
                if not csv_files:
                    raise ValueError(f"No CSV file found in ZIP: {file_path}")
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
                print_sys(f"Loaded {csv_files[0]} from {os.path.basename(file_path)}")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        df["__source_file__"] = os.path.basename(file_path)
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        return train_df.to_dict(orient="records")

    except Exception as e:
        print_sys(f"Error loading NEHRS data: {e}")
        return None
