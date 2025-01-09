import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .rond import getROND
from .vqa_rad import getVQA_RAD
from .pmc_vqa import getPMC_VQA
from .pubMedQA import getPubMedQA
from .medMCQA import getMedMCQA
from .medQA_USMLE import getMedQA_USMLE
from .liveQA_PREC_2017 import getLiveQA_PREC_2017
from .medicationQA import getMedicationQA
from .CT_RATE import getCT_RATE
from .LLaVA_Med import getLLaVA_Med
from .Path_VQA import getPath_VQA

class DataLoader(baseLoader.DataLoader):
    """A base data loader class for inference.

    Args:
        name (str): the dataset name.
        path (str): The path to save the data file
        print_stats (bool): Whether to print basic statistics of the dataset

    Attributes:
        trainset (list): a dict of the inference trainset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        testset (list): a dict of the inference testset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        valset (list): a dict of the inference valset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        alldata(dict): a dict of the whole inference dataset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        name (str): dataset name
        path (str): path to save and retrieve the dataset
        support_format (list<str>): format valid for current dataset
        support_subset (list<str>): subset valid for current dataset
    """

    def __init__(
        self,
        name,
        path="./data",
        print_stats=False,
    ):
        """
        Create a base dataloader object that each inference task dataloader class can inherit from.
        """
        
        self.name = name
        self.path = path

        self.trainset = None
        self.testset = None
        self.valset = None
        self.alldata = None
        self.support_format = []
        self.support_subset = []

        if self.name == "rond":
            datasets = getROND(self.path)
            self.alldata = datasets
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "vqa-rad":
            datasets = getVQA_RAD(self.path)
            self.alldata = datasets
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "pmc-vqa":
            datasets = getPMC_VQA(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif "pubmedqa" in self.name:
            subtitle = self.name.split("-")[1]
            datasets = getPubMedQA(self.path, subtitle)
            self.alldata = datasets
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "medmcqa":
            datasets = getMedMCQA(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif "medqa_usmle" in self.name:
            subtitle = self.name.split("-")[1]
            datasets = getMedQA_USMLE(self.path, subtitle)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif self.name == "LiveQA_PREC_2017":
            datasets = getLiveQA_PREC_2017(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif self.name == "MedicationQA":
            datasets = getMedicationQA(self.path)
            self.alldata = datasets
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "CT_RATE":
            datasets = getCT_RATE(self.path)
            self.alldata = datasets
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "LLaVA_Med":
            datasets = getLLaVA_Med(self.path)
            self.alldata = datasets
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "Path_VQA":
            datasets = getPath_VQA(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        else:
            raise ValueError(f"Dataset {self.name} is not supported.")

        if print_stats:
            self.print_stats()
