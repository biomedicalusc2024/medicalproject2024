import nibabel as nib
import nibabel.processing
from .model import *
from .utils import *


def JERS_preprocess(datasets, save_dir, img_size=96, 
                    ext_stage=5, reg_stage=5, gamma=10, beta=0.1):
    """
    datasets(list(pd.DataFrame)): each df in datasets should at least contain 2 columns: raw_img_path & dataset_name
    """
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = JERS(img_size, ext_stage, reg_stage, gamma, beta)

    model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "model_weights.pth")
    if not os.path.exists(model_path):
        # download checkpoints from cloud
        pass
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    JERS_save_dir = os.path.join(save_dir, "JERS")
    os.makedirs(JERS_save_dir, exist_ok=True)

    for dataset in datasets:
        try:
            for index, row in dataset.iterrows():
                image = read_and_resize(row["raw_img_path"])
                output = inference_jers(model, image)
        
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue


def read_and_resize(raw_img_path):
    raw_img = nib.load(raw_img_path)
    voxel_size = np.array(raw_img.shape) / np.array([96,96,96])
    voxel_size = np.round(voxel_size, 2)
    resampled_raw_img = nibabel.processing.resample_to_output(raw_img, voxel_size, order=1)
    resampled_raw_np = image_to_square_v2(resampled_raw_img.get_fdata(), 96)
    normed_raw_np = norm_to_0_1(resampled_raw_np)
    return normed_raw_np


def inference_jers(model, image):
    with torch.no_grad():
        inference_res = model(
            fixed_data,
            image,
            fixed_label_am,
            if_train=False
        )
