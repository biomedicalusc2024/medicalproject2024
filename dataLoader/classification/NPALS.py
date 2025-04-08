import os
import warnings
import pandas as pd
import zipfile

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

# NPALS dataset info
NPALS_YEAR = "2022"

NPALS_FILES = {
    "CSV": [
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Provider-CSV-Data-File.csv",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Services-User-CSV-Data-File.csv",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Provider-CSV-Data-File.csv",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Services-User-CSV-Data-File.csv"
    ],
    "Codebooks": [
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Provider-PUF-Codebook.xlsx",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Services-User-PUF-Codebook.xlsx",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Provider-PUF-Codebook.xlsx",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Services-User-PUF-Codebook.xlsx"
    ],
    "SAS": [
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Provider-SAS-Data-File.sas7bdat",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Provider-SAS-Input-Statements.sas",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Services-User-SAS-Data-File.sas7bdat",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Services-User-SAS-Input-Statements.sas",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Provider-SAS-Data-File.sas7bdat",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Provider-SAS-Input-Statements.sas",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Services-User-SAS-Data-File.sas7bdat",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Services-User-SAS-Input-Statements.sas",
    ],
    "STATA": [
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Provider-STATA-Data-File.dta",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Provider-STATA-Input-Statements.do",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Services-User-STATA-Data-File.dta",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Services-User-STATA-Input-Statements.do",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Provider-STATA-Data-File.dta",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Provider-STATA-Input-Statements.do",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Services-User-STATA-Data-File.dta",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Services-User-STATA-Input-Statements.do",
    ],
    "R": [
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Provider-R-Data-File.rds",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-ADSC-Provider-R-ReadMe-Document.R",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Provider-R-Data-File.Rds",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Provider-R-ReadMe-Document.R",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Services-User-R-Data-File.Rds",
        "https://ftp.cdc.gov/pub/health_statistics/nchs/datasets/npals/2022/2022-NPALS-RCC-Services-User-R-ReadMe-Document.R"
    ],
    "PDFs": [
        "https://www.cdc.gov/nchs/data/npals/2022-NPALS-RCC-Questionnaire-Community.pdf",
        "https://www.cdc.gov/nchs/data/npals/2022-NPALS-RCC-Questionnaire-Resident.pdf"
    ]
}

def getNPALS(path):
    year_path = os.path.join(path, "NPALS", NPALS_YEAR)

    for category, urls in NPALS_FILES.items():
        category_path = os.path.join(year_path, category)
        os.makedirs(category_path, exist_ok=True)

        for url in urls:
            file_name = url.split("/")[-1]
            file_path = os.path.join(category_path, file_name)

            if not os.path.exists(file_path):
                print_sys(f"Downloading NPALS file: {file_name}")
                download_file(url, file_path, category_path)
            else:
                print_sys(f"Found local file: {file_name}")

            if file_name.endswith(".csv"):
                _ = loadCSV(file_path)
            elif file_name.endswith(".zip"):
                _ = extractZip(file_path, category_path)
            # PDFs, Excel, STATA, SAS: just download and store

def loadCSV(file_path):
    try:
        df = pd.read_csv(file_path)
        df['__source_file__'] = os.path.basename(file_path)
        return df.to_dict(orient="records")
    except Exception as e:
        print_sys(f"Error loading CSV: {e}")
        return None

def extractZip(file_path, extract_to):
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            z.extractall(path=extract_to)
            print_sys(f"Extracted zip: {file_path}")
    except Exception as e:
        print_sys(f"Failed to extract zip: {e}")
