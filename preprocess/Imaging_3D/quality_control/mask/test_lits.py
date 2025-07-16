import os
import gdown
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .model import QualitySentinel
from .dataset import Clip_Rescale, LiTSMaskDataset
from .....dataLoader.segmentation import DataLoader as segLoader


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(FILE_PATH, 'label_embedding.pkl'), 'rb') as file:
    embedding_dict = pickle.load(file)

def download_default_model():
    model_url = "https://drive.google.com/uc?id=10K_D67vXIG8w41hTIhFXQjLRcXgExCRA"
    local_path = os.path.join(FILE_PATH, "../../../defaultData/3D_imaging/best_resnet50_model_40_samples.pth")
    local_path = os.path.abspath(local_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return local_path

    try:
        gdown.download(model_url, local_path, quiet=False)
        return local_path
    except:
        return local_path

def default_lits():
    # Hyperparameters
    model_name = 'resnet50'
    train_samples = 40
    epochs = 30
    batch_size = 128
    num_workers = 1
    learning_rate = 0.001
    weight_decay = 1e-4
    info_interval = 1
    eval_interval = 1
    # TRAIN_DATA_PATH = 'Quality_Sentinel_data_50samples/train'
    # VALID_DATA_PATH = 'Quality_Sentinel_data_50samples/val'
    MODEL_SAVE_PATH = download_default_model()
    if not os.path.exists(MODEL_SAVE_PATH):
        raise ValueError(f"Default model download false, please download manually and place it at {MODEL_SAVE_PATH}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preprocessing
    transform_ct = transforms.Compose([
        Clip_Rescale(min_val=-200, max_val=200),
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])

    transform_mask = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print('Loading LiTS data...')
    img_paths = segLoader(name="LiTS").get_data("df")
    lits_dataset = LiTSMaskDataset(img_paths, transform_ct, transform_mask, data_type="2d", mode='test', num_samples=10)
    lits_loader = DataLoader(lits_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    model = QualitySentinel(hidden_dim=50, backbone=model_name, embedding='text_embedding')
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)

    model.eval()
    pred_dices = []
    mask_names = []
    with torch.no_grad():
        for ct, pred_mask, gt_mask, dice, mask_class, mask_path in tqdm(lits_loader, desc="LiTs Inference", total=len(lits_loader)):
            ct, pred_mask, gt_mask, dice, mask_class = ct.to(device), pred_mask.to(device), gt_mask.to(device), dice.unsqueeze(1).to(device), mask_class.unsqueeze(1).to(device)
 
            text_embeddings = embedding_dict[mask_class[0].item()]  # [512]
            text_embeddings = text_embeddings  # [1, 512]
            text_embeddings = text_embeddings.to(device)

            predicted_dice = model(torch.cat((ct, pred_mask), dim=1), text_embeddings)
            pred_dices.append(predicted_dice.item())
            mask_names.append(mask_path[0])
            print(pred_dices)
            print(mask_names)
    
    print("\nResults Summary:")
    print(f"Average predicted Dice: {np.mean(pred_dices):.4f}")
    print("\nDetailed Results:")
    for name, dice in zip(mask_names, pred_dices):
        print(f"{os.path.basename(name)}: {dice:.4f}")

if __name__ == "__main__":
    default_lits()