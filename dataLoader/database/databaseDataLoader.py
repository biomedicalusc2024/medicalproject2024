import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .CORD19 import getCORD19, CORD19_SUBTITLE

SUPPORTED_DATASETS = [f"CORD19-{sub}" for sub in CORD19_SUBTITLE] + []

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

        if "CORD19" in self.name:
            subtitle = self.name.split("-")[1]
            datasets = getCORD19(self.path, subtitle)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        else:
            raise ValueError(f"Dataset {self.name} is not supported. Please select name in {SUPPORTED_DATASETS}.")

        if print_stats:
            self.print_stats()
