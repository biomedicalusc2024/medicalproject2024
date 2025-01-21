import os
import shutil
import datasets

from ..utils import print_sys, download_file


# tested by tjl 2025/1/20
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
            download_file(url, zip_path, img_path)
            shutil.move(os.path.join(img_path, "imgs"), imgs_path)
            shutil.rmtree(img_path)

        valid_list = []
        for f1 in os.listdir(imgs_path):
            f1_path = os.path.join(imgs_path, f1)
            if not os.path.isdir(f1_path):
                continue
            for f2 in os.listdir(f1_path):
                f2_path = os.path.join(f1, f2)
                valid_list.append(f2_path)

        print_sys("Loading files...")
        slake_train = slake["train"].to_pandas()
        slake_train = slake_train[slake_train["img_name"].isin(valid_list)]
        slake_train["img_name"] = slake_train["img_name"].apply(lambda x: os.path.join(imgs_path, x))
        train_dataset = slake_train.to_dict(orient='records')

        slake_test = slake["test"].to_pandas()
        slake_test = slake_test[slake_test["img_name"].isin(valid_list)]
        slake_test["img_name"] = slake_test["img_name"].apply(lambda x: os.path.join(imgs_path, x))
        test_dataset = slake_test.to_dict(orient='records')

        slake_val = slake["validation"].to_pandas()
        slake_val = slake_val[slake_val["img_name"].isin(valid_list)]
        slake_val["img_name"] = slake_val["img_name"].apply(lambda x: os.path.join(imgs_path, x))
        val_dataset = slake_val.to_dict(orient='records')

        return train_dataset, test_dataset, val_dataset

    except Exception as e:
        print_sys(f"error: {e}")
