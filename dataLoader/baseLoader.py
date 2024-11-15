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

    def get_data(self, format="df"):
        """
        Arguments:
            format (str, optional): the dataset format

        Returns:
            pd.DataFrame/dict/np.array: when format is df/dict/DeepPurpose

        Raises:
            AttributeError: format not supported
        """
        if format == "df":
            return pd.DataFrame({
                self.source_name + "_ID": self.source_idx,
                self.source_name: self.source,
                "Target": self.target,
            })
        elif format == "dict":
            return {
                self.source_name + "_ID": self.source_idx,
                self.source_name: self.source,
                "Target": self.target,
            }
        elif format == "DeepPurpose":
            return self.source, self.target
        else:
            raise AttributeError("Please use the correct format input")

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