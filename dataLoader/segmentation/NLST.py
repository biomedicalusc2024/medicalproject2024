import os
import requests
from tqdm import tqdm

# dataset not public, skipped

def getNLST(path):
    """
    Fetch NLST files and save them locally.

    Args:
        path (str): The directory to save the PDF files.
    """
    urls = {
        "Participant": "https://cdas.cancer.gov/files/download/mt68952kup/participant.dictionary.d040722.pdf",
        "Spiral CT Screening": "https://cdas.cancer.gov/files/download/7zx2ja0q07/sct_screening.dictionary.d040722.pdf",
        "PChest X-Ray Screening": "https://cdas.cancer.gov/files/download/7gk9qohj06/cxr_screening.dictionary.d040722.pdf",
        "Spiral CT Abnormalities": "https://cdas.cancer.gov/files/download/1md1y3rjsx/sct_abnormalities.dictionary.d040722.pdf",
        "PChest X-Ray Abnormalities": "https://cdas.cancer.gov/files/download/9ej1nd679i/cxr_abnormalities.dictionary.d040722.pdf",
        "Spiral CT Comparison Read Abnormalities": "https://cdas.cancer.gov/files/download/s4x07c3hld/sct_comparison_abnorm.dictionary.d040722.pdf",
        "Chest X-Ray Comparison Read Abnormalities": "https://cdas.cancer.gov/files/download/b1ple24jei/cxr_comparison_abnorm.dictionary.d040722.pdf",
        "Diagnostic Procedures": "https://cdas.cancer.gov/files/download/8n5cc7l96m/diagnostic_procedures.dictionary.d040722.pdf",
        "Medical Complications": "https://cdas.cancer.gov/files/download/ibaoocfmnu/medical_complications.dictionary.d040722.pdf",
        "Lung Cancer": "https://cdas.cancer.gov/files/download/9izn1ajpmo/lung_cancer.dictionary.d040722.pdf",
        "Treatment": "https://cdas.cancer.gov/files/download/3v0yhck457/treatment.dictionary.d040722.pdf",
        "Cause of Death": "https://cdas.cancer.gov/files/download/yqy974nfz5/cause_of_death.dictionary.d040722.pdf",
        "LSS Non-cancer Condition": "https://cdas.cancer.gov/files/download/9s4egdhdge/lss_noncanc_conditions.dictionary.d040722.pdf",
        "ACRIN Non-lung-cancer Condition": "https://cdas.cancer.gov/files/download/iywxup6oyb/acrin_noncanc_conditions.dictionary.d040722.pdf",
        "LSS HAQ": "https://cdas.cancer.gov/files/download/d5gw0py7ol/lss_haq.dictionary.d040722.pdf",
        "Spiral CT Image Information": "https://cdas.cancer.gov/files/download/c83pyhxxsw/sct_image_series.dictionary.d040722.pdf",
        "Pathology Image": "https://cdas.cancer.gov/files/download/rypdukrqoc/pathology_images.dictionary.d040722.pdf",
    }

    datasetLoad(urls=urls, path=path, datasetName="NLST_Files")


def datasetLoad(urls, path, datasetName):
    """
    Download the PDFs if not already available locally.

    Args:
        urls (dict): Dictionary of file names and their URLs.
        path (str): Base directory to save the files.
        datasetName (str): Name of the dataset.
    """
    datasetPath = os.path.join(path, datasetName)
    os.makedirs(datasetPath, exist_ok=True)

    total_files = len(urls)
    downloaded_files = 0

    for name, url in urls.items():
        file_name = f"{name}.pdf"
        file_path = os.path.join(datasetPath, file_name)

        # Download the file if not already present
        if not os.path.exists(file_path):
            print(f"Downloading {file_name} ({downloaded_files + 1}/{total_files})...")
            try:
                download_file(url, file_path)
                downloaded_files += 1
            except Exception as e:
                print(f"Failed to download {file_name}. Error: {e}")
        else:
            print(f"{file_name} already exists.")

  
    print(f"All files are saved to: {datasetPath}")


def download_file(url, file_path):
    """
    Download a file from the given URL and save it to the specified path.

    Args:
        url (str): URL of the file to download.
        file_path (str): Path to save the downloaded file.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses

    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, "wb") as file, tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))


