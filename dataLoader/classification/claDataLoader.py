import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .medMnist import getMedMnist

class DataLoader(baseLoader.DataLoader):
    """A base data loader class for segmentation.

    Args:
        name (str): the dataset name.
        path (str): The path to save the data file
        print_stats (bool): Whether to print basic statistics of the dataset

    Attributes:
        trainset (list): a list of the segmentation trainset ([source_path, target_path])
        testset (list): a list of the segmentation testset ([source_path, target_path])
        alldata(dict): a dict for all data preserving folder structure
        name (str): dataset name
        path (str): path to save and retrieve the dataset
    """

    def __init__(
        self,
        name,
        path="./data",
        print_stats=False,
    ):
        """
        Create a base dataloader object that each segmentation task dataloader class can inherit from.
        Raises:
            VauleError:
        """
        
        self.name = name
        self.path = path

        self.trainset = None
        self.testset = None
        self.valset = None
        self.alldata = None

        if "medmnist" in self.name:
            sub = self.name.split("-")[-1]
            datasets = getMedMnist(self.path, sub)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
        else:
            raise ValueError(f"Dataset {self.name} is not supported.")

        if print_stats:
            self.print_stats()

    def get_data(self, format="df", dataset="train"):
        """
        Arguments:
            format (str, optional): the returning dataset format, defaults to 'df'
            dataset (str, optional): which dataset to return, defaults to "training"

        Returns:
            pandas DataFrame/dict: a dataframe of a dataset/a dictionary for key information in the dataset

        Raises:
            AttributeError: Use the correct format input (df, dict, DeepPurpose)
        """

        # if format == "df":
        #     if dataset == "train":
        #         return pd.DataFrame(self.trainset, columns=['source', 'target'])
        #     elif dataset == "test":
        #         return pd.DataFrame(self.testset, columns=['source', 'target'])
        #     elif dataset == "val":
        #         return pd.DataFrame(self.valset, columns=['source', 'target'])
        #     else:
        #         raise AttributeError("Please select the dataset input in ['train', 'test', 'val']")
        if format == "dict":
            if dataset == "train":
                if self.trainset is not None:
                    return {
                        "source": self.trainset["source"],
                        "target": self.trainset["target"]
                    }
                else:
                    raise AttributeError("trainset is not allowed in current dataset")
            elif dataset == "test":
                if self.testset is not None:
                    return {
                        "source": self.testset["source"],
                        "target": self.testset["target"]
                    }
                else:
                    raise AttributeError("testset is not allowed in current dataset")
            elif dataset == "val":
                if self.valset is not None:
                    return {
                        "source": self.valset["source"],
                        "target": self.valset["target"]
                    }
                else:
                    raise AttributeError("valset is not allowed in current dataset")
            elif dataset == "all":
                pass
            else:
                raise AttributeError("Please select the dataset input in ['train', 'test', 'val', 'all']")
        elif format == "DeepPurpose":
            if dataset == "train":
                if self.trainset is not None:
                    return self.trainset
                else:
                    raise AttributeError("trainset is not allowed in current dataset")
            elif dataset == "test":
                if self.testset is not None:
                    return self.testset
                else:
                    raise AttributeError("testset is not allowed in current dataset")
            elif dataset == "val":
                if self.valset is not None:
                    return self.valset
                else:
                    raise AttributeError("valset is not allowed in current dataset")
            elif dataset == "all":
                pass
            else:
                raise AttributeError("Please select the dataset input in ['train', 'test', 'val', 'all']")
        else:
            raise AttributeError("Please select the format input in ['df', 'dict', 'DeepPurpose']")