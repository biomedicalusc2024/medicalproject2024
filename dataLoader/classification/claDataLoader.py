import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .medMnist import getMedMnist
from .rond import getROND
from .chestxrays import getChestXRays
from .cirrhosis import getCirrhosis
from .heartFailurePrediction import getHeartFailurePrediction
from .hepatitisCPrediction import getHepatitisCPrediction
from .ptb_xl import getPTB_XL
from .hoc import getHoC

class DataLoader(baseLoader.DataLoader):
    """A base data loader class for classification.

    Args:
        name (str): the dataset name.
        path (str): The path to save the data file
        print_stats (bool): Whether to print basic statistics of the dataset

    Attributes:
        trainset (list): a dict of the classification trainset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        testset (list): a dict of the classification testset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        valset (list): a dict of the classification valset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        alldata(dict): a dict of the whole classification dataset if exist({"source": [source_1, ...], "target": [target_1, ...]})
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
        Create a base dataloader object that each classification task dataloader class can inherit from.
        """
        
        self.name = name
        self.path = path

        self.trainset = None
        self.testset = None
        self.valset = None
        self.alldata = None
        self.support_format = []
        self.support_subset = []

        if "medmnist" in self.name:
            subtitle = self.name.split("-")[-1]
            datasets = getMedMnist(self.path, subtitle)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["dict", "DeepPurpose"]
            self.support_subset = ["train", "test", "val", "all"]
        elif self.name == "rond":
            datasets = getROND(self.path)
            self.alldata = datasets
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "chestxrays":
            datasets = getChestXRays(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "cirrhosis":
            datasets = getCirrhosis(self.path)
            self.alldata = datasets
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "heartFailurePrediction":
            datasets = getHeartFailurePrediction(self.path)
            self.alldata = datasets
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "hepatitisCPrediction":
            datasets = getHepatitisCPrediction(self.path)
            self.alldata = datasets
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "ptb-xl":
            datasets = getPTB_XL(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "dict", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "hoc":
            datasets = getHoC(self.path)
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