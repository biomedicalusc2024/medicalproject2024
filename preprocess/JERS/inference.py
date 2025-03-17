import os
from tqdm import tqdm
import nibabel as nib
import nibabel.processing

from .model import *
from .utils import norm_to_0_1, image_to_square_v2, save_nii_any


def JERS_preprocess(datasets, save_dir, img_size=96, 
                    ext_stage=5, reg_stage=5, gamma=10):
    """
    datasets(dict(name: pd.DataFrame)): each df in datasets should at least contain 1 columns: raw_img_path
    """
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = JERS(img_size, ext_stage, reg_stage, gamma)

    model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "model_state.pt")
    if not os.path.exists(model_path):
        # download checkpoints from cloud
        pass
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    fixed_data=torch.from_numpy(np.load(os.path.join(os.path.dirname(__file__), "checkpoints", "template_img_orig_96.npy"))).to(device).view(-1,1,img_size,img_size,img_size).float()
    fixed_label_am=torch.from_numpy(np.load(os.path.join(os.path.dirname(__file__), "checkpoints", "template_img_gm_mask_orig_96.npy"))).to(device).view(-1,1,img_size,img_size,img_size).float()

    JERS_save_dir = os.path.join(save_dir, "JERS")
    os.makedirs(JERS_save_dir, exist_ok=True)

    result_dict = {}
    for dataset_name, dataset in datasets.items():
        try:
            path_dict_list = []
            for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Processing images"):
                image = read_and_resize(row["raw_img_path"], device, img_size)
                output = inference_jers(model, image, fixed_data, fixed_label_am)
                dataset_path = os.path.join(JERS_save_dir, dataset_name)

                path_dict = {"index": index}
                path_dict["image"] = save_nii_any(image, os.path.join(dataset_path, "image"), index)
                path_dict["reginv_am"] = save_nii_any(output[6], os.path.join(dataset_path, "reginv_am"), index)
                path_dict["segpred_am"] = save_nii_any(torch.argmax(output[5], axis=1), os.path.join(dataset_path, "segpred_am"), index)

                path_dict["mask"] = []
                path_dict["striped"] = []
                for t in range(reg_stage):
                    path_dict["mask"].append(save_nii_any(output[0][t], os.path.join(dataset_path, "mask"), f"{index}_{t}"))
                    path_dict["striped"].append(save_nii_any(output[1][t], os.path.join(dataset_path, "striped"), f"{index}_{t}"))

                path_dict["warped"] = []
                for y in range(ext_stage):
                    path_dict["warped"].append(save_nii_any(output[2][t], os.path.join(dataset_path, "warped"), f"{index}_{y}"))

                path_dict_list.append(path_dict)
            result_dict[dataset_name] = path_dict_list
        
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue
    
    return result_dict


def read_and_resize(raw_img_path, device, img_size):
    raw_img = nib.load(raw_img_path)
    voxel_size = np.array(raw_img.shape) / np.array([img_size,img_size,img_size])
    voxel_size = np.round(voxel_size, 2)
    resampled_raw_img = nibabel.processing.resample_to_output(raw_img, voxel_size, order=1)
    resampled_raw_np = image_to_square_v2(resampled_raw_img.get_fdata(), img_size)
    normed_raw_np = norm_to_0_1(resampled_raw_np)
    normed_raw_np = torch.from_numpy(normed_raw_np).to(device).view(-1,1,img_size,img_size,img_size).float()
    return normed_raw_np


def inference_jers(model, image, fixed_data, fixed_label_am):
    with torch.no_grad():
        inference_res = model(
            fixed_data,
            image,
            fixed_label_am,
            if_train=False
        )
    return inference_res
