import os
import zipfile
import warnings
from tqdm import tqdm
import requests
from collections import defaultdict

warnings.filterwarnings("ignore")

from ..utils import print_sys

# need account and password, skipped

def getOCTA500(path):
    """
    Fetch OCTA Dataset, process, and return in the following format:
        rawdata (list): A list of all extracted file paths.
    """
    urls = {
        "Label": "https://ieee-dataport.s3.amazonaws.com/open/18078/Label.zip?response-content-disposition=attachment%3B%20filename%3D%22Label.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=42e1a83bc3ca73e000ff6a75195eb1ab1ca656737fd0a7d34dba9a6ec17ba9d5",
        "Code": "https://ieee-dataport.s3.amazonaws.com/open/18078/Code.zip?response-content-disposition=attachment%3B%20filename%3D%22Code.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=b6b3c218fc4b1a114bc0c2638885330ab60f0665c4d3204718f265b98796729f",
        "OCTA_3mm_part1": "https://ieee-dataport.s3.amazonaws.com/open/18078/OCTA_3mm_part1.zip?response-content-disposition=attachment%3B%20filename%3D%22OCTA_3mm_part1.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=c30919eb54dce0d7b4e88efe9f5027cd34471fc2f7f4c646c1e7f321da4c600a",
        "OCTA_3mm_part2": "https://ieee-dataport.s3.amazonaws.com/open/18078/OCTA_3mm_part2.zip?response-content-disposition=attachment%3B%20filename%3D%22OCTA_3mm_part2.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=1c2cf015fc8517489015f6858f8f6bec6c5fed2f40b337c32021fad48bd3571a",
        "OCTA_3mm_part3": "https://ieee-dataport.s3.amazonaws.com/open/18078/OCTA_3mm_part3.zip?response-content-disposition=attachment%3B%20filename%3D%22OCTA_3mm_part3.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=5e6c9702e918ee3311634dad704c2395f549fc0537c7fcd99bdbc7213186cb31",
        "OCTA_6mm_part1": "https://ieee-dataport.s3.amazonaws.com/open/18078/OCTA_6mm_part1.zip?response-content-disposition=attachment%3B%20filename%3D%22OCTA_6mm_part1.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=1b2cfd3af27d7cf8eb428e539939138d4fb70ba6b1f52a3988e2a2bd60e53f45",
        "OCTA_6mm_part2": "https://ieee-dataport.s3.amazonaws.com/open/18078/OCTA_6mm_part2.zip?response-content-disposition=attachment%3B%20filename%3D%22OCTA_6mm_part2.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=d624ee9c7cd030e09d62bd5856a48f55d4cb87557108ed94be6a987314595b48",
        "OCTA_6mm_part3": "https://ieee-dataport.s3.amazonaws.com/open/18078/OCTA_6mm_part3.zip?response-content-disposition=attachment%3B%20filename%3D%22OCTA_6mm_part3.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=15739bc3d9bb7d615cfab0a643c5d87405602e183205b78eaff1fb981aa076ca",
        "OCTA_6mm_part4": "https://ieee-dataport.s3.amazonaws.com/open/18078/OCTA_6mm_part4.zip?response-content-disposition=attachment%3B%20filename%3D%22OCTA_6mm_part4.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=5c986be90a07c993999c2eaad7d40c4db95f7c5bfa1902a7400ad6c22c2fc91d",
        "OCTA_6mm_part5": "https://ieee-dataport.s3.amazonaws.com/open/18078/OCTA_6mm_part5.zip?response-content-disposition=attachment%3B%20filename%3D%22OCTA_6mm_part5.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=d59c5c73dfd74066a4c64a949d276477e2f247998818b9334e4cb2ecdca56c70",
        "OCTA_6mm_part6": "https://ieee-dataport.s3.amazonaws.com/open/18078/OCTA_6mm_part6.zip?response-content-disposition=attachment%3B%20filename%3D%22OCTA_6mm_part6.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=9b0991e7919b4846b16a496cb032990aa9d4eb2a7b36d80c6e94b3874019fdc4",
        "OCTA_6mm_part7": "https://ieee-dataport.s3.amazonaws.com/open/18078/OCTA_6mm_part7.zip?response-content-disposition=attachment%3B%20filename%3D%22OCTA_6mm_part7.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=0c66958b1c834c31fd120d56e3507070374d4fd99b6e4dbd2b3e3612202d9300",
        "OCTA_6mm_part8": "https://ieee-dataport.s3.amazonaws.com/open/18078/OCTA_6mm_part8.zip?response-content-disposition=attachment%3B%20filename%3D%22OCTA_6mm_part8.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T005158Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=acd5fbb5db93ce0671528e50733e9c44876d5aa21fce56d6abe015ba606730ca",
    }

    rawdata = datasetLoad(urls=urls, path=path, datasetName="OCTA")

    if not rawdata:
        raise ValueError("Failed to fetch or process the OCTA Dataset.")

    return rawdata


def datasetLoad(urls, path, datasetName):
    """
    Download and process the dataset if not already available locally.

    Args:
        urls (dict): Dictionary of dataset names and their URLs.
        path (str): Base directory to save the files.
        datasetName (str): Name of the dataset.

    Returns:
        rawdata (list): List of all extracted file paths.
    """
    datasetPath = os.path.join(path, datasetName)
    rawdata = []

    os.makedirs(datasetPath, exist_ok=True)

    for task, url in urls.items():
        file_name = f"{task}.zip"
        file_path = os.path.join(datasetPath, file_name)

        # Download the file if not already present
        if not os.path.exists(file_path):
            print(f"Downloading {task}...")
            download_file(url, file_path)

        # Extract the file if it hasn't been extracted yet
        extracted_dir = os.path.join(datasetPath, task)
        if not os.path.exists(extracted_dir):
            print(f"Extracting {file_name}...")
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(path=extracted_dir)
                print(f"Extraction complete for {file_name}.")
            except zipfile.BadZipFile as e:
                print(f"Failed to extract {file_name}. Error: {e}")
                continue

        # Collect all extracted file paths
        for root, _, files in os.walk(extracted_dir):
            for file in files:
                rawdata.append(os.path.join(root, file))

    print(f"Total files collected: {len(rawdata)}")
    return rawdata


def download_file(url, file_path):
    """
    Download a file from the given URL and save it to the specified path.

    Args:
        url (str): URL of the file to download.
        file_path (str): Path to save the downloaded file.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status() 

    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, "wb") as file, tqdm(
        total=total_size, unit="iB", unit_scale=True, desc=file_path.split("/")[-1]
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))


def loadLocalFiles(path):
    """
    Process the local files into a flat list of file paths.
    """
    rawdata = []

    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            rawdata.append(file_path)

    print_sys(f"Total files collected: {len(rawdata)}")
    return rawdata
