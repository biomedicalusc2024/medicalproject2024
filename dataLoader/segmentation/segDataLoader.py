import warnings

warnings.filterwarnings("ignore")

from .. import baseLoader
from .ACDC import getACDC
from .BraTS import getBraTS
from .BUID import getBUID, BUID_SUBTITLE
from .CIR import getCIR, CIR_SUBTITLE
from .Kvasir import getKvasir
from .Pancreas import getPancreas
from .ISIC_2018 import getISIC_2018, ISIC_2018_SUBTITLE
from .ISIC_2019 import getISIC_2019
from .LA import getLA
from .LiTS import getLiTS
from .Hippo import getHippo
from .ChestXray import getChestXray
from .MSD import getMSD, MSD_SUBTITLE
# from .NLST import getNLST
# from .OCTA500 import getOCTA500
from .Covid_QU_EX import getCovid_QU_EX
from .CheXmask import getCheXmask
from .SIIM_ACR_Pneumothorax import getSIIM_ACR
from .CBIS_DDSM import getCBIS_DDSM
# from .BKAI_IGH_NeoPolyp import getBKAI_IGH

SUPPORTED_DATASETS = [f"BUID-{sub}" for sub in BUID_SUBTITLE]
SUPPORTED_DATASETS += [f"CIR-{sub}" for sub in CIR_SUBTITLE]
SUPPORTED_DATASETS += [f"ISIC_2018-{sub}" for sub in ISIC_2018_SUBTITLE]
SUPPORTED_DATASETS += [f"MSD-{sub}" for sub in MSD_SUBTITLE]
SUPPORTED_DATASETS += ["ACDC", "BraTS", "Kvasir", "Pancreas", "ISIC_2019", "LA", "LiTS", "Hippo", 
                       "ChestXray", "Covid_QU_EX", "SIIM_ACR", "CBIS_DDSM"]

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

        if self.name == "ACDC":
            datasets = getACDC(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "BraTS":
            datasets = getBraTS(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif "BUID" in self.name:
            subtitle = self.name.split("-")[-1]
            datasets = getBUID(self.path, subtitle)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif "CIR" in self.name: 
            subtitle = self.name.split("-")[-1]
            datasets = getCIR(self.path, subtitle)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "Kvasir": 
            datasets = getKvasir(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "Pancreas": 
            datasets = getPancreas(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif "ISIC_2018" in self.name: 
            subtitle = self.name.split("-")[-1]
            datasets = getISIC_2018(self.path, subtitle)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif self.name == "ISIC_2019":
            datasets = getISIC_2019(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "LA": 
            datasets = getLA(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "LiTS": 
            datasets = getLiTS(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "Hippo": 
            datasets = getHippo(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "ChestXray": 
            datasets = getChestXray(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif "MSD" in self.name: 
            subtitle = self.name.split("-")[-1]
            datasets = getMSD(self.path, subtitle)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        # elif self.name == "NLST": 
        #     datasets = getNLST(self.path)
        #     self.alldata = datasets
        #     self.support_format = ["df", "DeepPurpose"]
        #     self.support_subset = ["all"]
        # elif self.name == "OCTA500": 
        #     datasets = getOCTA500(self.path)
        #     self.alldata = datasets
        #     self.support_format = ["df", "DeepPurpose"]
        #     self.support_subset = ["all"]
        elif self.name == "Covid_QU_EX": 
            datasets = getCovid_QU_EX(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.valset = datasets[2]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "validation", "all"]
        elif self.name == "CheXmask": 
            datasets = getCheXmask(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        elif self.name == "SIIM_ACR": 
            datasets = getSIIM_ACR(self.path)
            self.trainset = datasets[0]
            self.testset = datasets[1]
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["train", "test", "all"]
        elif self.name == "CBIS_DDSM": 
            datasets = getCBIS_DDSM(self.path)
            self.alldata = datasets
            self.support_format = ["df", "DeepPurpose"]
            self.support_subset = ["all"]
        # elif self.name == "BKAI_IGH": 
        #     datasets = getBKAI_IGH(self.path)
        #     self.alldata = datasets
        #     self.support_format = ["df", "DeepPurpose"]
        #     self.support_subset = ["all"]
        else:
            raise ValueError(f"Dataset {self.name} is not supported. Please select name in {SUPPORTED_DATASETS}.")

        if print_stats:
            self.print_stats()
