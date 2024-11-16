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
        """
        pass

    def get_data(self, format="df", dataset="train"):
        """
        Arguments:
            format (str, optional): the dataset format
            dataset (str, optional): which dataset to return, defaults to "training"

        Returns:
            pd.DataFrame/dict/list: when format is df/dict/DeepPurpose

        Raises:
            AttributeError: format not supported
        """
        if format == "df":
            if dataset == "train":
                return pd.DataFrame(self.trainset, columns=['source', 'target'])
            elif dataset == "test":
                return pd.DataFrame(self.testset, columns=['source', 'target'])
            elif dataset == "all":
                return pd.DataFrame(self.trainset+self.testset, columns=['source', 'target'])
            else:
                raise AttributeError("Please select the dataset input in ['train', 'test', 'all']")
        elif format == "dict":
            if dataset == "train":
                return {
                    "source": [item[0] for item in self.trainset],
                    "target": [item[1] for item in self.trainset]
                }
            elif dataset == "test":
                return {
                    "source": [item[0] for item in self.testset],
                    "target": [item[1] for item in self.testset]
                }
            elif dataset == "all":
                return {
                    "source": [item[0] for item in self.trainset+self.testset],
                    "target": [item[1] for item in self.trainset+self.testset]
                }
            else:
                raise AttributeError("Please select the dataset input in ['train', 'test', 'all']")
        elif format == "DeepPurpose":
            if dataset == "train":
                return self.trainset
            elif dataset == "test":
                return self.testset
            elif dataset == "all":
                return self.trainset+self.testset
            else:
                raise AttributeError("Please select the dataset input in ['train', 'test', 'all']")
        else:
            raise AttributeError("Please select the format input in ['df', 'dict', 'DeepPurpose']")

    def __len__(self):
        """
        get number of data points

        Returns:
            int: number of data points
        """
        return len(self.get_data(format="df"))
    
    def print_stats(self):
        """
        print statistics
        """
        print("statistics")