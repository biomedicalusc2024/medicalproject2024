import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader

class DataLoader(baseLoader.DataLoader):
    """A base data loader class for segmentation.

    Args:
        name (str): the dataset name.
        path (str): The path to save the data file
        print_stats (bool): Whether to print basic statistics of the dataset

    Attributes:
        source (Pandas Series): a list of the segmentation sources
        source_idx (Pandas Series): a list of the segmentation sources index
        source_name (Pandas Series): a list of the segmentation sources names
        target (Pandas Series): a list of the segmentation target
        name (str): dataset name
        path (str): path to save and retrieve the dataset
    """

    def __init__(
        self,
        name,
        path,
        print_stats,
    ):
        """
        Create a base dataloader object that each segmentation task dataloader class can inherit from.
        Raises:
            VauleError:
        """
        
        self.name = name
        self.path = path
        self.source = None
        self.source_idx = None
        self.source_name = None
        self.target = None

    def get_data(self, format="df"):
        """
        Arguments:
            format (str, optional): the returning dataset format, defaults to 'df'

        Returns:
            pandas DataFrame/dict: a dataframe of a dataset/a dictionary for key information in the dataset

        Raises:
            AttributeError: Use the correct format input (df, dict, DeepPurpose)
        """
        # to do
        pass