import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .acdc import getACDC
from .BraTS import getBraTS
from .buid import getBUID
from .cir import getCIR  
from .kvasir import getKvasir
from .pancreas import getPancreas
from .la import getLA
from .lits import getLiTS
from .hippo import getHippo
from .chestXray import getChestXray
class DataLoader(baseLoader.DataLoader):
    """A base data loader class for segmentation.

    Args:
        name (str): the dataset name.
        path (str): The path to save the data file
        print_stats (bool): Whether to print basic statistics of the dataset

    Attributes:
        trainset (list): a list of the segmentation trainset ([source_path, target_path])
        testset (list): a list of the segmentation testset ([source_path, target_path])
        valtset (list): a list of the segmentation valtset ([source_path, target_path])
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
            ValueError:
        """
        
        self.name = name
        self.path = path

        self.trainset = None
        self.testset = None
        self.valset = None
        self.alldata = None

        if self.name == "acdc":
            rawData = getACDC(self.path)
            self.alldata = rawData[0]
            self.testset = rawData[1]
            self.trainset = rawData[2]
        elif self.name == "brats":
            rawData = getBraTS(self.path)
            self.alldata = rawData[0]
            self.testset = rawData[1]
            self.trainset = rawData[2]
        elif self.name == "buid":
            rawData = getBUID(self.path)
            self.alldata = rawData[0]
            self.testset = rawData[1]
            self.trainset = rawData[2]
        elif self.name == "cir": 
            rawData = getCIR(self.path)
            self.alldata = rawData[0]
        elif self.name == "kvasir": 
            rawData = getKvasir(self.path)
            self.alldata = rawData[0]
        elif self.name == "pancreas": 
            rawData = getPancreas(self.path)
            self.alldata = rawData[0]
        elif self.name == "la": 
            rawData = getLA(self.path)
            self.alldata = rawData[0]
        elif self.name == "lits": 
            rawData = getLiTS(self.path)
            self.alldata = rawData[0]
        elif self.name == "hippo": 
            rawData = getHippo(self.path)
            self.alldata = rawData[0]
        elif self.name == "chestXray": 
            rawData = getChestXray(self.path)
            self.alldata = rawData[0]
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

        if format == "df":
            if dataset == "train":
                if self.trainset is not None:
                    return pd.DataFrame(self.trainset, columns=['source', 'target'])
                else:
                    raise AttributeError("trainset is not allowed in current dataset")
            elif dataset == "test":
                if self.testset is not None:
                    return pd.DataFrame(self.testset, columns=['source', 'target'])
                else:
                    raise AttributeError("testset is not allowed in current dataset")
            elif dataset == "val":
                if self.valset is not None:
                    return pd.DataFrame(self.valset, columns=['source', 'target'])
                else:
                    raise AttributeError("valset is not allowed in current dataset")
            elif dataset == "all":
                if self.alldata is not None:
                    return pd.DataFrame(self.alldata, columns=['source', 'target'])
                else:
                    all_data = []
                    for subset in [self.trainset, self.testset, self.valset]:
                        if subset is not None:
                            all_data = all_data + subset
                    return pd.DataFrame(all_data, columns=['source', 'target'])
            else:
                raise AttributeError("Please select the dataset input in ['train', 'test', 'val', 'all']")
        elif format == "dict":
            if dataset == "train":
                if self.trainset is not None:
                    return {
                        "source": [item[0] for item in self.trainset],
                        "target": [item[1] for item in self.trainset]
                    }
                else:
                    raise AttributeError("trainset is not allowed in current dataset")
            elif dataset == "test":
                if self.testset is not None:
                    return {
                        "source": [item[0] for item in self.testset],
                        "target": [item[1] for item in self.testset]
                    }
                else:
                    raise AttributeError("testset is not allowed in current dataset")
            elif dataset == "all":
                if self.alldata is not None:
                    return {
                        "source": [item[0] for item in self.alldata],
                        "target": [item[1] for item in self.alldata]
                    }
                else:
                    all_data = []
                    for subset in [self.trainset, self.testset, self.valset]:
                        if subset is not None:
                            all_data = all_data + subset
                    return {
                        "source": [item[0] for item in all_data],
                        "target": [item[1] for item in all_data]
                    }
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
            elif dataset == "all":
                if self.alldata is not None:
                    return self.alldata
                else:
                    all_data = []
                    for subset in [self.trainset, self.testset, self.valset]:
                        if subset is not None:
                            all_data = all_data + subset
                    return all_data
            else:
                raise AttributeError("Please select the dataset input in ['train', 'test', 'val', 'all']")
        else:
            raise AttributeError("Please select the format input in ['df', 'dict', 'DeepPurpose']")