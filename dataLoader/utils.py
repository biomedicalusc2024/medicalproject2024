import os
import sys
import rarfile
import tarfile
import zipfile
import requests
import numpy as np
from tqdm import tqdm

def print_sys(s):
    """system print
    Args:
        s (str): the string to print
    """
    print(s, flush=True, file=sys.stderr)


def download_file(url, destination, extractionPath=None):
    try:
        headers = {
            'Accept': '*/*',
             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        response = requests.get(url, stream=True, headers=headers)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(destination, "wb") as file:
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

            if extractionPath:
                if "zip" in destination:
                    with zipfile.ZipFile(destination, "r") as z:
                        z.extractall(extractionPath)
                    print_sys("Extraction complete.")
                    os.remove(destination)
                elif "rar" in destination:
                    with rarfile.RarFile(destination) as rf:
                        rf.extractall(extractionPath)
                    print_sys("Extraction complete.")
                    os.remove(destination)
                elif "tar" in destination:
                    if "gz" in destination:
                        with tarfile.open(destination, 'r:gz') as tar:
                            tar.extractall(extractionPath)
                        print_sys("Extraction complete.")
                        os.remove(destination)
                    else:
                        with tarfile.open(destination, 'r') as tar:
                            tar.extractall(extractionPath)
                        print_sys("Extraction complete.")
                        os.remove(destination)

        else:
            print_sys("url error")
    except Exception as e:
        print_sys(f"error: {e}")


def dice_coef(y_true, y_pred, smooth=1):
    """
    Calculate the Dice coefficient between two binary arrays.

    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array.
    y_pred : array_like
        Predicted binary mask array.
    smooth : float, optional
        Smoothing factor to avoid division by zero.

    Returns:
    -------
    float
        Dice coefficient between `y_true` and `y_pred`.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat)

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_accuracy(y_true, y_pred, threshold=0.5, smooth=1):
    """
    Calculate the Dice accuracy between two arrays by thresholding predictions.

    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array.
    y_pred : array_like
        Predicted probability mask array (values between 0 and 1).
    threshold : float, optional
        Threshold to binarize `y_pred`.
    smooth : float, optional
        Smoothing factor to avoid division by zero.

    Returns:
    -------
    float
        Dice accuracy between `y_true` and the binarized `y_pred`.
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    return dice_coef(y_true, y_pred_bin, smooth)

def dice_loss(y_true, y_pred, smooth=1):
    """
    Calculate the Dice loss between two binary arrays.

    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array.
    y_pred : array_like
        Predicted binary mask array.
    smooth : float, optional
        Smoothing factor to avoid division by zero.

    Returns:
    -------
    float
        Dice loss between `y_true` and `y_pred`.
    """
    dice = dice_coef(y_true, y_pred, smooth)
    loss = 1 - dice
    return loss

def iou(y_true, y_pred, smooth=1):
    """
    Calculate the Intersection over Union (IoU) between two binary arrays.

    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array.
    y_pred : array_like
        Predicted binary mask array.
    smooth : float, optional
        Smoothing factor to avoid division by zero.

    Returns:
    -------
    float
        IoU between `y_true` and `y_pred`.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection

    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score

def hausdorff_distance(y_true, y_pred):
    """
    Calculate the Hausdorff distance between two binary arrays.

    This metric requires identifying the set of coordinates where each mask 
    has values equal to 1, and then computing the maximum distance of a point in 
    one set to the closest point in the other set.

    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array.
    y_pred : array_like
        Predicted binary mask array.

    Returns:
    -------
    float
        Hausdorff distance between `y_true` and `y_pred`.
    """
    true_points = np.argwhere(y_true > 0)
    pred_points = np.argwhere(y_pred > 0)

    if len(true_points) == 0 and len(pred_points) == 0:
        # Both empty masks, distance is 0
        return 0.0
    if len(true_points) == 0:
        # if true mask is empty, distance defined by any point in pred
        return np.max([np.min(np.sqrt(np.sum((p - tp)**2))) 
                       for p in pred_points for tp in pred_points]) if len(pred_points) > 0 else 0.0
    if len(pred_points) == 0:
        # if pred mask is empty, distance defined by any point in true
        return np.max([np.min(np.sqrt(np.sum((t - tt)**2))) 
                       for t in true_points for tt in true_points]) if len(true_points) > 0 else 0.0

    # get distances from each true point to the nearest pred point
    distances_true_to_pred = []
    for t in true_points:
        distances = np.sqrt(np.sum((pred_points - t)**2, axis=1))
        distances_true_to_pred.append(np.min(distances))

    # get distances from each pred point to the nearest true point
    distances_pred_to_true = []
    for p in pred_points:
        distances = np.sqrt(np.sum((true_points - p)**2, axis=1))
        distances_pred_to_true.append(np.min(distances))

    #   hd = max(h(A,B), h(B,A))
    hd = max(np.max(distances_true_to_pred), np.max(distances_pred_to_true))
    return hd
