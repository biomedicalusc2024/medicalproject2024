import os
import sys
import rarfile
import tarfile
import zipfile
import requests
import numpy as np
from tqdm import tqdm


def print_sys(s):
    """system print
    Args:
        s (str): the string to print
    """
    print(s, flush=True, file=sys.stderr)


def download_file(url, destination, extractionPath=None):
    try:
        headers = {
            'Accept': '*/*',
             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        print_sys("Downloading...")
        response = requests.get(url, stream=True, headers=headers)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(destination, "wb") as file:
                if total_size == 0:
                    pbar = None
                else:
                    pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        if pbar:
                            pbar.update(len(chunk))
                if pbar:
                    pbar.close()
            print_sys("Download complete.")

            if extractionPath:
                if "zip" in destination:
                    with zipfile.ZipFile(destination, "r") as z:
                        z.extractall(extractionPath)
                    print_sys("Extraction complete.")
                    os.remove(destination)
                elif "rar" in destination:
                    with rarfile.RarFile(destination) as rf:
                        rf.extractall(extractionPath)
                    print_sys("Extraction complete.")
                    os.remove(destination)
                elif "tar" in destination:
                    if "gz" in destination:
                        with tarfile.open(destination, 'r:gz') as tar:
                            tar.extractall(extractionPath)
                        print_sys("Extraction complete.")
                        os.remove(destination)
                    elif "bz" in destination:
                        with tarfile.open(destination, "r:bz2") as tar:
                            tar.extractall(extractionPath)
                        print_sys("Extraction complete.")
                        os.remove(destination)
                    else:
                        with tarfile.open(destination, 'r') as tar:
                            tar.extractall(extractionPath)
                        print_sys("Extraction complete.")
                        os.remove(destination)

        else:
            print_sys("url error")
    except Exception as e:
        print_sys(f"error: {e}")
