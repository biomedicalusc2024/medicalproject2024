import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .rond import getROND
from .vqa_rad import getVQA_RAD
from .pmc_vqa import getPMC_VQA

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
        else:
            raise ValueError(f"Dataset {self.name} is not supported.")

        if print_stats:
            self.print_stats()
