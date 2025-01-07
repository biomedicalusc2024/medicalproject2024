import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .rond import getROND
from .ddiextraction2013 import getDDIEtraction2013
from .sourceData import getSourceData

class DataLoader(baseLoader.DataLoader):
    """A base data loader class for NER(named entity recognition).

    Args:
        name (str): the dataset name.
        path (str): The path to save the data file
        print_stats (bool): Whether to print basic statistics of the dataset

    Attributes:
        trainset (list): a dict of the NER trainset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        testset (list): a dict of the NER testset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        valset (list): a dict of the NER valset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        alldata(dict): a dict of the whole NER dataset if exist({"source": [source_1, ...], "target": [target_1, ...]})
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
        Create a base dataloader object that each NER task dataloader class can inherit from.
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
        elif self.name == "ddiextracion2013":
            datasets = getDDIEtraction2013(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "sourceData":
            datasets = getSourceData(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        else:
            raise ValueError(f"Dataset {self.name} is not supported.")

        if print_stats:
            self.print_stats()
