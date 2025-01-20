import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .ROND import getROND
from .VQA_RAD import getVQA_RAD
from .PMC_VQA import getPMC_VQA
from .MedMCQA import getMedMCQA
from .MedQA_USMLE import getMedQA_USMLE, MedQA_USMLE_SUBTITLE
from .LiveQA_PREC_2017 import getLiveQA_PREC_2017
from .MedicationQA import getMedicationQA
from .CT_RATE import getCT_RATE
from .LLaVA_Med import getLLaVA_Med
from .Path_VQA import getPath_VQA
from .WSI_VQA import getWSI_VQA
from .PubMedQA import getPubMedQA, PubMedQA_SUBTITLE

SUPPORTED_DATASETS = [f"PubMedQA-{sub}" for sub in PubMedQA_SUBTITLE]
SUPPORTED_DATASETS += [f"MedQA_USMLE-{sub}" for sub in MedQA_USMLE_SUBTITLE]
SUPPORTED_DATASETS += ["ROND", "VQA_RAD", "PMC_VQA", "MedMCQA", "LiveQA_PREC_2017",
                       "MedicationQA", "LLaVA_Med", "Path_VQA", "WSI_VQA"]

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
        elif self.name == "VQA_RAD":
            datasets = getVQA_RAD(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "PMC_VQA":
            datasets = getPMC_VQA(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif "PubMedQA" in self.name:
            subtitle = self.name.split("-")[1]
            datasets = getPubMedQA(self.path, subtitle)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "MedMCQA":
            datasets = getMedMCQA(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif "MedQA_USMLE" in self.name:
            subtitle = self.name.split("-")[1]
            datasets = getMedQA_USMLE(self.path, subtitle)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif self.name == "LiveQA_PREC_2017":
            datasets = getLiveQA_PREC_2017(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif self.name == "MedicationQA":
            datasets = getMedicationQA(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        # elif self.name == "CT_RATE":
        #     datasets = getCT_RATE(self.path)
        #     self.alldata = datasets
        #     self.support_format = ["df", "DeepPurpose"]
        #     self.support_subset = ["all"]
        elif self.name == "LLaVA_Med":
            datasets = getLLaVA_Med(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "Path_VQA":
            datasets = getPath_VQA(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif self.name == "WSI_VQA":
            datasets = getWSI_VQA(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        else:
            raise ValueError(f"Dataset {self.name} is not supported. Please select name in {SUPPORTED_DATASETS}.")

        if print_stats:
            self.print_stats()
