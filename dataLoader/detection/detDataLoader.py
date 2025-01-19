import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .ChestXRays import getChestXRays

SUPPORTED_DATASETS = ["ChestXRays"]

class DataLoader(baseLoader.DataLoader):
    """A base data loader class for detection.

    Args:
        name (str): the dataset name.
        path (str): The path to save the data file
        print_stats (bool): Whether to print basic statistics of the dataset

    Attributes:
        trainset (list): a dict of the detection trainset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        testset (list): a dict of the detection testset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        valset (list): a dict of the detection valset if exist({"source": [source_1, ...], "target": [target_1, ...]})
        alldata(dict): a dict of the whole detection dataset if exist({"source": [source_1, ...], "target": [target_1, ...]})
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
        Create a base dataloader object that each detection task dataloader class can inherit from.
        """
        
        self.name = name
        self.path = path

        self.trainset = None
        self.testset = None
        self.valset = None
        self.alldata = None
        self.support_format = []
        self.support_subset = []

        if self.name == "ChestXRays":
            datasets = getChestXRays(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        else:
            raise ValueError(f"Dataset {self.name} is not supported. Please select name in {SUPPORTED_DATASETS}.")

        if print_stats:
            self.print_stats()
