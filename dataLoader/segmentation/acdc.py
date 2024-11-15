import sys
import warnings

warnings.filterwarnings("ignore")

from . import segDataLoader

class ACDC(segDataLoader.DataLoader):
    """
    Data loader class to load ACDC datasets in segmentation task. 
    More info: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb/folder/637218e573e9f0047faa00fc

    Args:
        name (str): the dataset name.
        path (str, optional):
            The path to save the data file, defaults to './data'
        print_stats (bool, optional):
            Whether to print basic statistics of the dataset, defaults to False
    """

    def __init__(
        self,
        name,
        path="./data",
        print_stats=None,
    ):
        """Create a ACDC dataloader object."""
        super().__init__(
            name,
            path,
            print_stats,
        )
        if print_stats:
            self.print_stats()
        print("Done!", flush=True, file=sys.stderr)
