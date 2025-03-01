import os
import warnings
import datasets

warnings.filterwarnings("ignore")

from ..utils import print_sys

CORD19_SUBTITLE = ["metadata", "fulltext", "embeddings"]


# tested by tjl 2025/1/17
def getCORD19(path, subtitle):
    if subtitle not in CORD19_SUBTITLE:
        raise AttributeError(f'Please enter dataset name in CORD19-subset format and select the subsection of CORD19 in {CORD19_SUBTITLE}')
    try:
        data_path = os.path.join(path, "CORD19", subtitle)
        if os.path.exists(data_path):
            ds = datasets.load_from_disk(data_path)
        else:
            ds = datasets.load_dataset("allenai/cord19", subtitle, trust_remote_code=True)
            ds.save_to_disk(data_path)
            
        df_train = ds["train"].to_pandas()

        dataset_train = df_train.to_dict(orient='records')

        return dataset_train

    except Exception as e:
        print_sys(f"error: {e}")