import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .ROCO import getROCO, ROCO_SUBTITLE
from .IUXray import getIUXray
from .PMC_OA import getPMC_OA

SUPPORTED_DATASETS = [f"ROCO-{sub}" for sub in ROCO_SUBTITLE]
SUPPORTED_DATASETS += ["IUXray", "PMC_OA"]

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
