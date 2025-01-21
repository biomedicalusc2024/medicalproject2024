import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .TREC import getTREC
from .PubMed import getPubMed
from .MeQSum import getMeQSum

SUPPORTED_DATASETS = ["PubMed", "MeQSum", "TREC"]

class DataLoader(baseLoader.DataLoader):
    """
    refer to baseLoader
    """

    def __init__(
        self,
        name,
        path="./data",
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

        if self.name == "PubMed":
            datasets = getPubMed(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "MeQSum":
            datasets = getMeQSum(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "TREC":
            datasets = getTREC(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        else:
            raise ValueError(f"Dataset {self.name} is not supported. Please select name in {SUPPORTED_DATASETS}.")

        if print_stats:
            self.print_stats()
