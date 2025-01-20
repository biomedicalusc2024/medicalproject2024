import os
import warnings
import datasets

warnings.filterwarnings("ignore")

from ..utils import print_sys


# need access token on huggingface, temporarily skipped
def getCT_RATE(path):
    try:
        data_path = os.path.join(path, "CT_RATE")
        if os.path.exists(data_path):
            ds = datasets.load_from_disk(data_path)
        else:
            ds = datasets.load_dataset("ibrahimhamamci/CT-RATE", trust_remote_code=True)
            ds.save_to_disk(data_path)
            
        df = ds["train"].to_pandas()
        dataset = df.to_dict(orient='records')
        return dataset

    except Exception as e:
        print_sys(f"error: {e}")