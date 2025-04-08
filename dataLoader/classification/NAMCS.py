import os
import warnings
import pandas as pd
import zipfile

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

try:
    import pyreadstat  # for .sas7bdat and .sav
except ImportError:
    pyreadstat = None

try:
    import pyreadr  # for .rds
except ImportError:
    pyreadr = None

# NAMCS dataset info
NAMCS_YEAR = "2022"

NAMCS_FILES = {
    "CSV": [
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Provider-CSV-Data-File.csv",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Services-User-CSV-Data-File.csv",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Provider-CSV-Data-File.csv",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Services-User-CSV-Data-File.csv"
    ],
    "SAS": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NAMCS_HC/2023/namcshc2023_sas.sas7bdat",
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NAMCS_HC/2023/namcshc2023_sas.sas",
    ],
    "STATA": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NAMCS_HC/2023/namcshc2023_stata.dta",
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NAMCS_HC/2023/namcshc2023_stata.do",
    ],
    "R": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NAMCS_HC/2023/namcshc2023_r.rds",
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NAMCS_HC/2023/namcshc2023_r.R",
    ],
    "PDFs": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NAMCS_HC/2023/2023-NAMCS-HC-Component-Tech-Doc.pdf",
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NAMCS_HC/2023/2023-NAMCS-HC-Component-PUF-Codebook.pdf"
    ]
}

def getNAMCS(path):
    year_path = os.path.join(path, "NAMCS", NAMCS_YEAR)

    for category, urls in NAMCS_FILES.items():
        category_path = os.path.join(year_path, category)
        os.makedirs(category_path, exist_ok=True)

        for url in urls:
            file_name = url.split("/")[-1]
            file_path = os.path.join(category_path, file_name)

            if not os.path.exists(file_path):
                print_sys(f"Downloading NAMCS file: {file_name}")
                download_file(url, file_path, category_path)
            else:
                print_sys(f"Found local file: {file_name}")

            _ = loadFile(file_path)

def loadFile(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".tsv":
            df = pd.read_csv(file_path, sep="\t")
        elif ext == ".dta":
            df = pd.read_stata(file_path)
        elif ext == ".sas7bdat":
            if pyreadstat:
                df, _ = pyreadstat.read_sas7bdat(file_path)
            else:
                print_sys("pyreadstat not installed. Skipping SAS file.")
                return None
        elif ext == ".xpt":
            df = pd.read_sas(file_path, format="xport")
        elif ext == ".rds":
            if pyreadr:
                result = pyreadr.read_r(file_path)
                df = next(iter(result.values()))
            else:
                print_sys("pyreadr not installed. Skipping RDS file.")
                return None
        elif ext in [".txt", ".dat"]:
            df = pd.read_fwf(file_path)
        elif ext == ".zip":
            extractZip(file_path, os.path.dirname(file_path))
            return None
        else:
            return None

        df["__source_file__"] = os.path.basename(file_path)
        print_sys(f"Loaded: {os.path.basename(file_path)} with {len(df)} rows")
        return df.to_dict(orient="records")

    except Exception as e:
        print_sys(f"Error loading {file_path}: {e}")
        return None

def extractZip(file_path, extract_to):
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            z.extractall(path=extract_to)
            print_sys(f"Extracted zip: {file_path}")
    except Exception as e:
        print_sys(f"Failed to extract zip: {e}")
