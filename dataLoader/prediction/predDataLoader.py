import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .NHIS import getNHIS
from .MEPS import getMEPS

SUPPORTED_DATASETS = ["NHIS", "MEPS"]

class DataLoader(baseLoader.DataLoader):
    """
    refer to baseLoader
    """

    def __init__(
        self,
        name,
        path="./data",
        variables=None,
        task=None,
        print_stats=False,
    ):
        self.name = name
        self.path = path

        self.trainset = None
        self.testset = None
        self.valset = None
        self.alldata = None
        self.support_format = []
        self.support_subset = []

        if self.name == "NHIS":
            datasets = getNHIS(self.path, variables, task)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "MEPS":
            datasets = getMEPS(self.path, variables, task)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        else:
            raise ValueError(f"Dataset {self.name} is not supported. Please select name in {SUPPORTED_DATASETS}.")

        if print_stats:
            self.print_stats()
