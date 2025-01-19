import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .ROCO import getROCO, ROCO_SUBTITLE
from .IUXray import getIUXray
from .PMC_OA import getPMC_OA

SUPPORTED_DATASETS = [f"ROCO-{sub}" for sub in ROCO_SUBTITLE] + ["IUXray", "PMC_OA"]

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
        self.support_format = []
        self.support_subset = []

        if "ROCO" in self.name:
            subtitle = self.name.split("-")[-1]
            datasets = getROCO(self.path, subtitle)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif self.name == "IUXray":
            datasets = getIUXray(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "PMC_OA":
            datasets = getPMC_OA(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        else:
            raise ValueError(f"Dataset {self.name} is not supported. Please select name in {SUPPORTED_DATASETS}.")

        if print_stats:
            self.print_stats()
