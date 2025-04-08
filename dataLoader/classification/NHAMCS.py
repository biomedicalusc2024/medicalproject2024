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

# NHAMCS dataset info
NHAMCS_YEAR = "2022"

NHAMCS_FILES = {
    "SAS": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/sas/",
    ],
    "SPSS": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/spss/",
    ],
    "text": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/readme2021.txt",
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/sas/readme2021-ed-sas.txt",
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/spss/readme2021-ed-spss.txt",
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/stata/readme2021-ed-stata.txt",
    ],
    "PDFs": [
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/doc21-ed-508.pdf",
    ]
}

def getNHAMCS(path):
    year_path = os.path.join(path, "NHAMCS", NHAMCS_YEAR)

    for category, urls in NHAMCS_FILES.items():
        category_path = os.path.join(year_path, category)
        os.makedirs(category_path, exist_ok=True)

        for url in urls:
            file_name = url.split("/")[-1]
            file_path = os.path.join(category_path, file_name)

            if not os.path.exists(file_path):
                print_sys(f"Downloading NHAMCS file: {file_name}")
                download_file(url, file_path, category_path)
            else:
                print_sys(f"Found local file: {file_name}")

            _ = loadFile(file_path)

def loadFile(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".tsv":
            df = pd.read_csv(file_path, sep="\t")
        elif ext == ".dta":
            df = pd.read_stata(file_path)
        elif ext == ".sas7bdat":
            if pyreadstat:
                df, _ = pyreadstat.read_sas7bdat(file_path)
            else:
                print_sys("pyreadstat not installed. Skipping SAS file.")
                return None
        elif ext == ".sav":
            if pyreadstat:
                df, _ = pyreadstat.read_sav(file_path)
            else:
                print_sys("pyreadstat not installed. Skipping SPSS file.")
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
            # Attempt to parse only if it's a table
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                sample = f.read(1024)
                if "," in sample or "\t" in sample:
                    df = pd.read_csv(file_path, sep=None, engine="python")
                elif any(char.isdigit() for char in sample[:100]):
                    df = pd.read_fwf(file_path)
                else:
                    print_sys(f"Text file stored (not parsed): {file_path}")
                    return None
        elif ext == ".pdf":
            print_sys(f"PDF file stored: {file_path}")
            return None
        elif ext == ".zip":
            extractZip(file_path, os.path.dirname(file_path))
            return None
        else:
            print_sys(f"Unsupported file type: {file_path}")
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
