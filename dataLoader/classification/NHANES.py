import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

# Dataset index organized by year and category
NHANES_INDEX = {
    "2021-2023": {
        "Demographics": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt",
        "Dietary": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR1IFF_L.xpt", # only cover one for now
        "Examination": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR1IFF_L.xpt", # only cover one for now
        "Lab": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR1IFF_L.xpt", # only cover one for now
        "Questionnaire": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/AUQ_L.xpt", # only cover one for now
    },
    "2017-2018": {
        "Demographics": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
        "Dietary": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DR1IFF_J.xpt", # only cover one for now
        "Examination": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/AUX_J.xptpt", # only cover one for now
        "Lab": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ALB_CR_J.xpt", # only cover one for now
        "Questionnaire": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ALB_CR_J.xpt", # only cover one for now
    }
}

def getNHANES(path, year="2017-2018", category="Demographics"):
    dataset_url = NHANES_INDEX.get(year, {}).get(category, None)
    if not dataset_url:
        print_sys(f"Dataset not found for year={year}, category={category}")
        return None, None

    datasetPath = os.path.join(path, "NHANES", year.replace("-", "_"), category)
    os.makedirs(datasetPath, exist_ok=True)

    file_name = dataset_url.split("/")[-1]
    file_path = os.path.join(datasetPath, file_name)

    if not os.path.exists(file_path):
        print_sys(f"Downloading NHANES file: {file_name}")
        download_file(dataset_url, file_path, datasetPath)
    else:
        print_sys(f"Found local file: {file_name}")

    return loadLocalFile(file_path)


def loadLocalFile(file_path):
    try:
        if file_path.endswith(".XPT"):
            df = pd.read_sas(file_path, format="xport", encoding="utf-8")
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format")

        df['__source_file__'] = os.path.basename(file_path)

        # Optional: simulate train/test split
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        train = train_df.to_dict(orient="records")
        test = test_df.to_dict(orient="records")

        return train, test
    except Exception as e:
        print_sys(f"Failed to load file: {e}")
        return None, None


# Helper function to list available NHANES datasets
def list_nhanes_datasets():
    print("Available NHANES datasets:")
    for year, categories in NHANES_INDEX.items():
        print(f"  {year}:")
        for category in categories:
            print(f"    - {category}")

