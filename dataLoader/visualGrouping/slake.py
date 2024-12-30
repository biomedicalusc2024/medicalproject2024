import os
import shutil
import zipfile
import datasets
import requests
from tqdm import tqdm

from ..utils import print_sys

def getSLAKE(path):
    try:
        data_path = os.path.join(path, "SLAKE")
        if os.path.exists(data_path):
            slake = datasets.load_from_disk(data_path)
        else:
            slake = datasets.load_dataset("BoKelvin/SLAKE", trust_remote_code=True)
            slake.save_to_disk(data_path)
        
        img_path = os.path.join(data_path, "img")
        imgs_path = os.path.join(data_path, "imgs")
        if not os.path.exists(imgs_path):
            url = "https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip?download=true"
            zip_path = os.path.join(data_path, "imgs.zip")
            print_sys("Downloading...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))

                with open(zip_path, "wb") as file:
                    if total_size == 0:
                        pbar = None
                    else:
                        pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))
                    if pbar:
                        pbar.close()
                print_sys("Download complete.")

                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(img_path)
                print_sys("Extraction complete.")
                os.remove(zip_path)

                shutil.move(os.path.join(img_path, "imgs"), imgs_path)
                shutil.rmtree(img_path)

        print_sys("Loading files...")
        source_cols = ['img_name', 'location', 'modality', 'base_type', 'question', 'qid', 'content_type', 'triple', 'img_id', 'q_lang']
        target_cols = ['answer', 'answer_type']

        slake_train = slake["train"].to_pandas()
        slake_train["img_name"] = slake_train["img_name"].apply(lambda x: os.path.join(imgs_path, x))
        train_dataset = {
            "source": slake_train[source_cols].to_numpy().tolist(),
            "target": slake_train[target_cols].to_numpy().tolist()
        }

        slake_test = slake["test"].to_pandas()
        slake_test["img_name"] = slake_test["img_name"].apply(lambda x: os.path.join(imgs_path, x))
        test_dataset = {
            "source": slake_test[source_cols].to_numpy().tolist(),
            "target": slake_test[target_cols].to_numpy().tolist()
        }

        slake_val = slake["validation"].to_pandas()
        slake_val["img_name"] = slake_val["img_name"].apply(lambda x: os.path.join(imgs_path, x))
        val_dataset = {
            "source": slake_val[source_cols].to_numpy().tolist(),
            "target": slake_val[target_cols].to_numpy().tolist()
        }

        return train_dataset, test_dataset, val_dataset

    except Exception as e:
        print_sys(f"error: {e}")
