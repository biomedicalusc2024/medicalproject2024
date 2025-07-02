import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .HoC import getHoC
from .ROND import getROND
from .PTB_XL import getPTB_XL
from .Cirrhosis import getCirrhosis
from .ChestXRays import getChestXRays
from .CheXpert_small import getCheXpert_small
from .StrokePrediction import getStrokePrediction
from .HepatitisCPrediction import getHepatitisCPrediction
from .HeartFailurePrediction import getHeartFailurePrediction
from .NHANES import getNHANES
from .NSFG import getNSFG
from .NEHRS import getNEHRS
from .NPALS import getNPALS
from .NAMCS import getNAMCS
from .NHAMCS import getNHAMCS
from .IS_A import getIS_A, IS_A_SUBTITLE
from .MedMnist import getMedMnist, MedMnist_SUBTITLE
from .NHIS import getNHIS
from .MEPS import getMEPS

SUPPORTED_DATASETS = [f"MedMnist-{sub}" for sub in MedMnist_SUBTITLE]
SUPPORTED_DATASETS += [f"IS_A-{sub}" for sub in IS_A_SUBTITLE]
SUPPORTED_DATASETS += ["ROND", "ChestXRays", "CheXpert_small", "Cirrhosis", "HeartFailurePrediction", 
                       "HepatitisCPrediction", "PTB_XL", "HoC", "StrokePrediction", "NHIS", "MEPS"]

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

        if "MedMnist" in self.name:
            subtitle = self.name.split("-")[-1]
            datasets = getMedMnist(self.path, subtitle)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif self.name == "ROND":
            datasets = getROND(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "ChestXRays":
            datasets = getChestXRays(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "CheXpert_small":
            datasets = getCheXpert_small(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "Cirrhosis":
            datasets = getCirrhosis(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "HeartFailurePrediction":
            datasets = getHeartFailurePrediction(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "HepatitisCPrediction":
            datasets = getHepatitisCPrediction(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "PTB_XL":
            datasets = getPTB_XL(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "HoC":
            datasets = getHoC(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "StrokePrediction":
            datasets = getStrokePrediction(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "NHANES":
            datasets = getNHANES(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "NSFG":
            datasets = getNHIS(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "NAMCS":
            datasets = getNHIS(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "NHAMCS":
            datasets = getNHIS(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "NEHRS":
            datasets = getNHIS(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "NPALS":
            datasets = getNHIS(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif "IS_A" in self.name:
            subtitle = self.name.split("-")[-1]
            datasets = getIS_A(self.path, subtitle)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "NHIS":
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
