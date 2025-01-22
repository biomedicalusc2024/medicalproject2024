import os
import shutil
import zipfile
import warnings
import subprocess

warnings.filterwarnings("ignore")

from ..utils import print_sys


# tested by tjl 2025/1/22
def getHoC(path):
    urls = ["https://github.com/sb895/Hallmarks-of-Cancer/archive/refs/heads/master.zip"]
    return datasetLoad(urls=urls, path=path, datasetName="HoC")


def datasetLoad(urls, path, datasetName):
    try:
        dataPath = os.path.join(path, datasetName)
        get_source(urls[0], dataPath)
        return loadLocalFiles(dataPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    sourcePath = os.path.join(path, "text")
    targetPath = os.path.join(path, "labels")
    id_list = os.listdir(sourcePath)
    source_list = [[os.path.join(sourcePath, s)] for s in id_list]
    target_list = [[os.path.join(targetPath, s)] for s in id_list]
    dataset = [{"text_path":text, "label_path":label} for text,label in zip(source_list,target_list)]
    return dataset


def get_source(source_url, target_path):
    if not (os.path.exists(os.path.join(target_path, "labels")) and os.path.exists(os.path.join(target_path, "text"))):
        os.makedirs(target_path, 0o755, exist_ok=True)
        if os.path.exists(os.path.join(target_path, "Hallmarks-of-Cancer-master")):
            shutil.rmtree(os.path.join(target_path, "Hallmarks-of-Cancer-master"))
        zip_filepath = os.path.join(target_path,"master.zip")
        subprocess.call(['wget', '-P', target_path, source_url])
        try:
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(target_path)
        except Exception as e:
            print(f"An error occurred: {e}")
        os.remove(zip_filepath)
        shutil.move(os.path.join(target_path, "Hallmarks-of-Cancer-master", "labels"), target_path)
        shutil.move(os.path.join(target_path, "Hallmarks-of-Cancer-master", "text"), target_path)
        shutil.rmtree(os.path.join(target_path, "Hallmarks-of-Cancer-master"))