import os
import re
import warnings

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/18
def getROND(path):
    urls = ["https://raw.githubusercontent.com/Mayo-Clinic-RadOnc-Foundation-Models/Radiation-Oncology-NLP-Database/main/3-Named%20Entity%20Recognition%20(NER)/Use-this-Wenxiong-New%20Named%20Entity%20Recognition%20Examples.txt"]
    return datasetLoad(urls=urls, path=path, datasetName="ROND")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(datasetPath, "ROND_NER.txt")
        if os.path.exists(datasetFile):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetFile)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], datasetFile)
            return loadLocalFiles(datasetFile)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    parse_res = parse_txt(path)
    return parse_res


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
    return {
        "sentence": sentence,
        "tokens_list": tokens_list,
        "ner_tags_list": ner_tags_list,
    }