"""
This file contains a base data loader object that specific one can inherit from. 
"""

import pandas as pd
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore")


class DataLoader:
    """
    base data loader class that contains functions shared by almost all data loader classes.
    """

    def __init__(self):
        """
        empty data loader class, to be overwritten
        basic attributes: (alldata can be none when any of train, test, val is not)
            trainset (list): a dict of the NER trainset if exist({"source": [source_1, ...], "target": [target_1, ...]})
            testset (list): a dict of the NER testset if exist({"source": [source_1, ...], "target": [target_1, ...]})
            valset (list): a dict of the NER valset if exist({"source": [source_1, ...], "target": [target_1, ...]})
            alldata(dict): a dict of the whole dataset if exist({"source": [source_1, ...], "target": [target_1, ...]})

            here all source item like source_1 should be a list containing all features dataset provide that can be used and target item similarly.
            we can add a explanation on both source and target about the meaning of each position in source and target item in print_stats or something else

            name (str): dataset name
            path (str): path to save and retrieve the dataset
            support_format (list<str>): format valid for current dataset
            support_subset (list<str>): subset valid for current dataset
        """
        self.trainset = None
        self.testset = None
        self.valset = None
        self.alldata = None
        self.support_format = []
        self.support_subset = []

    def get_data(self, format="dict", dataset="all"):
        """
        Arguments:
            format (str, optional): the dataset format
            dataset (str, optional): which dataset to return, defaults to "all"

        Returns:
            dict/pd.DataFrame/list: when format is dict/df/DeepPurpose

        Raises:
            AttributeError: format not supported
            AttributeError: dataset not supported
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
        elif dataset == "validation":
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

    def __len__(self):
        """
        get number of data points

        Returns:
            int: number of data points
        """
        return len(self.get_data(format="DeepPurpose", dataset="all"))
    
    def print_stats(self):
        """
        print statistics
        """
        print("statistics")