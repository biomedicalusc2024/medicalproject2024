import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .rond import getROND

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
        else:
            raise ValueError(f"Dataset {self.name} is not supported.")

        if print_stats:
            self.print_stats()

    def get_data(self, format="dict", dataset="all"):
        """
        Arguments:
            format (str, optional): the returning dataset format, defaults to 'dict'
            dataset (str, optional): which dataset to return, defaults to "all"

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