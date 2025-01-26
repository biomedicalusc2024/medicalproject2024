import os
import warnings

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/24
def getCovid_QU_EX(path):
    urls = ["https://www.kaggle.com/api/v1/datasets/download/anasmohammedtahir/covidqu"]
    return datasetLoad(urls=urls, path=path, datasetName="Covid_QU_EX")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        zipFile = os.path.join(datasetPath, "raw.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(datasetPath, exist_ok=True)
            download_file(urls[0], zipFile, datasetPath)
            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    infectionPath = os.path.join(path, "Infection Segmentation Data", "Infection Segmentation Data")
    lungPath = os.path.join(path, "Lung Segmentation Data", "Lung Segmentation Data")
    path_dict = {"infection":infectionPath, "lung":lungPath}

    datasets = {"Train":[], "Test":[], "Val":[]}
    for k,v in path_dict.items():
        for subset in ["Train", "Test", "Val"]:
            for specy in ["COVID-19", "Non-COVID", "Normal"]:
                images = os.path.join(v,subset,specy,"images")
                infections = os.path.join(v,subset,specy,"infection masks")
                lungs = os.path.join(v,subset,specy,"lung masks")
                for fn in os.listdir(images):
                    image_fn = os.path.join(images, fn)
                    infection_fn = os.path.join(infections, fn)
                    lung_fn = os.path.join(lungs, fn)
                    if k == "infection":
                        if os.path.exists(image_fn) and os.path.exists(infection_fn) and os.path.exists(lung_fn):
                            ele = {
                                "title": k,
                                "subtitle": specy,
                                "image_path": image_fn,
                                "infection_mask_path": infection_fn,
                                "lung_mask_path": lung_fn,
                            }
                            datasets[subset].append(ele)
                    elif k == "lung":
                        if os.path.exists(image_fn) and os.path.exists(lung_fn):
                            ele = {
                                "title": k,
                                "subtitle": specy,
                                "image_path": image_fn,
                                "lung_mask_path": lung_fn,
                            }
                            datasets[subset].append(ele)

    return datasets["Train"], datasets["Test"], datasets["Val"]