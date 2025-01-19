import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/17
def getPMC_OA(path):
    urls = [
        'https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/pmc_oa.jsonl?download=true',
        'https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/images.zip?download=true',
    ]
    return datasetLoad(urls, path=path, datasetName="PMC_OA")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        imgPath = os.path.join(datasetPath,'images')
        if os.path.exists(imgPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(os.path.join(datasetPath,'images'), exist_ok=True)

            download_file(urls[0], os.path.join(datasetPath, "pmc_oa.jsonl"))
            download_file(urls[1], os.path.join(datasetPath, "images.zip"), imgPath)

            return loadLocalFiles(datasetPath)
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    df = pd.read_json(os.path.join(path, "pmc_oa.jsonl"), lines=True)
    img_folder = os.path.join(path, "images", "caption_T060_filtered_top4_sep_v0_subfigures")
    imgs = os.listdir(img_folder)
    df = df[df['image'].isin(imgs)]
    df["image"] = df["image"].apply(lambda x: img_folder+"/"+x)
    dataset = df.to_dict(orient='records')
    return dataset