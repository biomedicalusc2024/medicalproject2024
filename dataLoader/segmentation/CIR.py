import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

CIR_SUBTITLE = ["LIDC", "LUNGx"]


# tested by tjl 2025/1/23
def getCIR(path, subtitle):
    urls = {
        "CIRDataset_LCSR": "https://zenodo.org/records/6762573/files/CIRDataset_LCSR.tar.bz2?download=1",
        # "CIRDataset_npy_for_cnn": "https://zenodo.org/records/6762573/files/CIRDataset_npy_for_cnn.tar.bz2?download=1",
        # "CIRDataset_pickle_for_voxel2mesh": "https://zenodo.org//records/6762573/files/CIRDataset_pickle_for_voxel2mesh.tar.bz2?download=1",
        # "pretrained_model_mesh_encoder": "https://zenodo.org//records/6762573/files/pretrained_model-mesh+encoder.tar.bz2?download=1",
        # "pretrained_model_meshonly": "https://zenodo.org//records/6762573/files/pretrained_model-meshonly.tar.bz2?download=1"
    }
    if subtitle not in CIR_SUBTITLE:
        raise AttributeError(f"Please enter dataset name in CIR-subset format and select the subsection of CIR in {CIR_SUBTITLE}")

    return datasetLoad(urls=urls, subtitle=subtitle, path=path, datasetName="CIR")

def datasetLoad(urls, subtitle, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath, subtitle)
        else:
            os.makedirs(datasetPath, exist_ok=True)

            for key, url in urls.items():
                file_name = f"{key}.tar.bz"
                file_path = os.path.join(datasetPath, file_name)
                extracted_dir = os.path.join(datasetPath, key)
                download_file(url, file_path, extracted_dir)

            return loadLocalFiles(datasetPath, subtitle)
    
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path, subtitle):
    base_path = os.path.join(path, "CIRDataset_LCSR", "DATA")
    LIDC_path = os.path.join(base_path, "LIDC_spiculation")
    LUNGx_path = os.path.join(base_path, "LUNGx_spiculation")
    df_LIDC = pd.read_csv(os.path.join(LIDC_path, "LIDC.csv"))
    df_LUNGx = pd.read_csv(os.path.join(LUNGx_path, "LUNGx.csv"))

    if subtitle == "LIDC":
        subpaths = [p for p in os.listdir(LIDC_path) if os.path.isdir(os.path.join(LIDC_path, p))]
        df_LIDC = df_LIDC[df_LIDC["PID"].isin(subpaths)]
        df_LIDC["PATH"] = df_LIDC["PID"].apply(lambda x: os.path.join(LIDC_path, x))
        dataset = df_LIDC.to_dict(orient='records')
    
    elif subtitle == "LUNGx":
        subpaths = [p for p in os.listdir(LUNGx_path) if os.path.isdir(os.path.join(LUNGx_path, p))]
        df_LUNGx = df_LUNGx[df_LUNGx["PID"].isin(subpaths)]
        df_LUNGx["PATH"] = df_LUNGx["PID"].apply(lambda x: os.path.join(LUNGx_path, x))
        dataset = df_LUNGx.to_dict(orient='records')

    return dataset
