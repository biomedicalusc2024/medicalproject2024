import os
import re
import requests
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys

def getROND(path):
    url = "https://raw.githubusercontent.com/Mayo-Clinic-RadOnc-Foundation-Models/Radiation-Oncology-NLP-Database/main/3-Named%20Entity%20Recognition%20(NER)/Use-this-Wenxiong-New%20Named%20Entity%20Recognition%20Examples.txt"
    return datasetLoad(url=url, path=path, datasetName="ROND")


def datasetLoad(url, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(datasetPath, "ROND_NER.txt")
        if os.path.exists(datasetFile):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetFile)
        else:
            print_sys("Downloading...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                os.makedirs(datasetPath, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(datasetFile, "wb") as file:
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
                return loadLocalFiles(datasetFile)
            else:
                print_sys("Connection error, please check the internet.")
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    parse_res = parse_txt(path)
    return {
        "source": [ele[0] for ele in parse_res],
        "target": [ele[1] for ele in parse_res],
    }


def parse_txt(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        entry_lines = ""
        for line in file:
            cleaned_line = line.strip("\ufeff")
            cleaned_line = cleaned_line.strip()
            if re.match(r"^\d+\.", cleaned_line):
                if entry_lines != "":
                    result.append(process_entry(entry_lines))
                    entry_lines = cleaned_line
                else:
                    entry_lines = cleaned_line
            else:
                entry_lines += cleaned_line
        result.append(process_entry(entry_lines))
    return result


def process_entry(entry_lines):
    entry_lines_split = entry_lines.split(": ")
    sentence = entry_lines_split[1].split("'")[0]
    tokens_str = entry_lines_split[2].split("'")[0]
    tokens_list = tokens_str[1:-1].split(", ")
    ner_tags_str = entry_lines_split[3]
    ner_tags_list = ner_tags_str[1:-1].split(", ")
    ner_tags_list = [re.split(r"\s*\(\s*|\s*\)\s*",ele)[:-1] for ele in ner_tags_list]
    return [[sentence, tokens_list], ner_tags_list]