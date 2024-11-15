import sys
import warnings

warnings.filterwarnings("ignore")

from . import segDataLoader

def getACDC():
    """
    fetch ACDC data, process and return in following format:
        source (Pandas Series): a list of the segmentation sources
        source_idx (Pandas Series): a list of the segmentation sources index
        source_name (Pandas Series): a list of the segmentation sources names
        target (Pandas Series): a list of the segmentation target
    """
    # to do
    pass