import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .ROND import getROND
from .SourceData import getSourceData
from .DDIEtraction2013 import getDDIEtraction2013

SUPPORTED_DATASETS = ["ROND", "DDIEtraction2013", "SourceData"]

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

        if self.name == "ROND":
            datasets = getROND(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "DDIEtraction2013":
            datasets = getDDIEtraction2013(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "SourceData":
            datasets = getSourceData(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        else:
            raise ValueError(f"Dataset {self.name} is not supported. Please select name in {SUPPORTED_DATASETS}.")

        if print_stats:
            self.print_stats()
