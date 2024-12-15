import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .acdc import getACDC
from .brats import getBraTS
from .buid import getBUID
from .cir import getCIR  
from .kvasir import getKvasir
from .pancreas import getPancreas
from .isic_2018 import getISIC_2018
from .isic_2019 import getISIC_2019
from .la import getLA
from .lits import getLiTS
from .hippo import getHippo
from .chestXray import getChestXray
from .msd import getMSD
from .nlst import getNLST
from .octa500 import getOCTA
from .covid_qu_ex import getCovid_QU_EX
from .chexmask import getCheXmask
from .siim_acr_pneumothorax import getSIIM_ACR
from .cbis_ddsm import getCBIS_DDSM
from .bkai_igh_neopolyp import getBKAI_IGH
class DataLoader(baseLoader.DataLoader):
    """A base data loader class for segmentation.

    Args:
        name (str): the dataset name.
        path (str): The path to save the data file
        print_stats (bool): Whether to print basic statistics of the dataset

    Attributes:
        trainset (list): a list of the segmentation trainset ([source_path, target_path])
        testset (list): a list of the segmentation testset ([source_path, target_path])
        valtset (list): a list of the segmentation valtset ([source_path, target_path])
        alldata(dict): a dict for all data preserving folder structure
        name (str): dataset name
        path (str): path to save and retrieve the dataset
    """

    def __init__(
        self,
        name,
        path="./data",
        print_stats=False,
    ):
        """
        Create a base dataloader object that each segmentation task dataloader class can inherit from.
        """
        
        self.name = name
        self.path = path

        self.trainset = None
        self.testset = None
        self.valset = None
        self.alldata = None
        self.support_format = []
        self.support_subset = []

        if self.name == "acdc":
            rawData = getACDC(self.path)
            self.alldata = rawData[0]
            self.testset = rawData[1]
            self.trainset = rawData[2]
        elif self.name == "brats":
            rawData = getBraTS(self.path)
            self.alldata = rawData[0]
            self.testset = rawData[1]
            self.trainset = rawData[2]
        elif self.name == "buid":
            rawData = getBUID(self.path)
            self.alldata = rawData[0]
            self.testset = rawData[1]
            self.trainset = rawData[2]
        elif self.name == "cir": 
            rawData = getCIR(self.path)
            self.alldata = rawData[0]
        elif self.name == "kvasir": 
            rawData = getKvasir(self.path)
            self.alldata = rawData[0]
        elif self.name == "pancreas": 
            rawData = getPancreas(self.path)
            self.alldata = rawData[0]
        elif self.name == "isic2018": 
            rawData = getISIC_2018(self.path)
            self.alldata = rawData[0]
        elif self.name == "isic2019": 
            rawData = getISIC_2019(self.path)
            self.alldata = rawData[0]
        elif self.name == "la": 
            rawData = getLA(self.path)
            self.alldata = rawData[0]
        elif self.name == "lits": 
            rawData = getLiTS(self.path)
            self.alldata = rawData[0]
        elif self.name == "hippo": 
            rawData = getHippo(self.path)
            self.alldata = rawData[0]
        elif self.name == "chestXray": 
            rawData = getChestXray(self.path)
            self.alldata = rawData[0]
        elif self.name == "bkaiigh": 
            rawData = getBKAI_IGH(self.path)
            self.alldata = rawData[0]
        elif self.name == "msd": 
            rawData = getMSD(self.path)
            self.alldata = rawData[0]
        elif self.name == "nlst": 
            rawData = getNLST(self.path)
            self.alldata = rawData[0]
        elif self.name == "octa": 
            rawData = getOCTA(self.path)
            self.alldata = rawData[0]
        elif self.name == "covidqu": 
            rawData = getCovid_QU_EX(self.path)
            self.alldata = rawData[0]
        elif self.name == "chexmask": 
            rawData = getCheXmask(self.path)
            self.alldata = rawData[0]
        elif self.name == "siimacr": 
            rawData = getSIIM_ACR(self.path)
            self.alldata = rawData[0]
        elif self.name == "cbisddsm": 
            rawData = getCBIS_DDSM(self.path)
            self.alldata = rawData[0]
        else:
            raise ValueError(f"Dataset {self.name} is not supported.")

        if print_stats:
            self.print_stats()

    def get_data(self, format="dict", dataset="all"):
        """
        Arguments:
            format (str, optional): the returning dataset format, defaults to 'df'
            dataset (str, optional): which dataset to return, defaults to "training"

        Returns:
            pandas DataFrame/dict: a dataframe of a dataset/a dictionary for key information in the dataset
        """
        if format not in self.support_format:
            raise AttributeError(f"{format} is not supported for current dataset, Please select the format input in {self.support_format}")
        
        if dataset not in self.support_subset:
            raise AttributeError(f"{dataset} is not supported for current dataset, Please select the dataset input in {self.support_subset}")

        return self._get_data(format, dataset)

    
    def _get_data(self, format, dataset):
        source = []
        target = []
        if dataset == "train":
            source = self.trainset["source"]
            target = self.trainset["target"]
        elif dataset == "test":
            source = self.testset["source"]
            target = self.testset["target"]
        elif dataset == "val":
            source = self.valset["source"]
            target = self.valset["target"]
        elif dataset == "all":
            if self.alldata is not None:
                source = self.alldata["source"]
                target = self.alldata["target"]
            else:
                if self.trainset is not None:
                    source = source + self.trainset["source"]
                    target = target + self.trainset["target"]
                if self.testset is not None:
                    source = source + self.testset["source"]
                    target = target + self.testset["target"]
                if self.valset is not None:
                    source = source + self.valset["source"]
                    target = target + self.valset["target"]
                if (source==[]) or (target==[]):
                    raise ValueError("No data in current dataset")
        else:
            raise AttributeError(f"{dataset} is not supported for current dataset, Please select the dataset input in {self.support_subset}")

        if format == "df":
            return pd.DataFrame({
                "source": source,
                "target": target
            })
        elif format == "dict":
            return {
                "source": source,
                "target": target
            }
        elif format == "DeepPurpose":
            return [[s,t] for s,t in zip(source, target)]
        else:
            raise AttributeError(f"{format} is not supported for current dataset, Please select the format input in {self.support_format}")