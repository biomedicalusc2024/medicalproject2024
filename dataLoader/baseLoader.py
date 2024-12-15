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
            alldata(dict): a dict of the whole NER dataset if exist({"source": [source_1, ...], "target": [target_1, ...]})
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
        pass

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