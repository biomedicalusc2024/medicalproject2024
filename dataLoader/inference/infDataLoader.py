import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .ROND import getROND
from .BioNLI import getBioNLI
from .NHIS import getNHIS
from .MEPS import getMEPS

SUPPORTED_DATASETS = ["ROND", "BioNLI", "NHIS", "MEPS"]

class DataLoader(baseLoader.DataLoader):
    """
    refer to baseLoader
    variables: list of variable names for NHIS and MEPS datasets
    task: int, index of target variable for NHIS and MEPS datasets
    """

    def __init__(
        self,
        name,
        path="./data",
        variables=None,
        task=None,
        print_stats=False,
    ):
        self.name = name
        self.path = path

        self.trainset = None
        self.testset = None
        self.valset = None
        self.alldata = None
        self.support_format = []
        self.support_subset = []

        if self.name == "ROND":
            datasets = getROND(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "BioNLI":
            datasets = getBioNLI(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif self.name == "NHIS":
            datasets = getNHIS(self.path, variables, task)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "MEPS":
            datasets = getMEPS(self.path, variables, task)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        else:
            raise ValueError(f"Dataset {self.name} is not supported. Please select name in {SUPPORTED_DATASETS}.")

        if print_stats:
            self.print_stats()
