import os
import cv2
import sys
import rarfile
import tarfile
import zipfile
import requests
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.SA_Score import sascorer
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                             f1_score, roc_curve, auc, confusion_matrix)
"""
A set of utility functions to support various metrics (segmentation, classification,
ranking, chemistry-based metrics, etc.), plus download and extraction helpers.

- Dice, IoU, Hausdorff Distance (segmentation metrics)
- Classification metrics (accuracy, recall, precision, F1, AUC, confusion matrix)
- Download and extraction of files (zip, rar, tar.gz)


"""
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
        return 0.0
    if len(true_points) == 0:
        return np.max([np.min(np.sqrt(np.sum((p - tp)**2))) 
                       for p in pred_points for tp in pred_points]) if len(pred_points) > 0 else 0.0
    if len(pred_points) == 0:
        return np.max([np.min(np.sqrt(np.sum((t - tt)**2))) 
                       for t in true_points for tt in true_points]) if len(true_points) > 0 else 0.0

    distances_true_to_pred = []
    for t in true_points:
        distances = np.sqrt(np.sum((pred_points - t)**2, axis=1))
        distances_true_to_pred.append(np.min(distances))

    distances_pred_to_true = []
    for p in pred_points:
        distances = np.sqrt(np.sum((true_points - p)**2, axis=1))
        distances_pred_to_true.append(np.min(distances))

    hd = max(np.max(distances_true_to_pred), np.max(distances_pred_to_true))
    return hd

def classification_metrics_sklearn(y_true, y_pred, threshold=0.5):
    """
    Provide dictionary of accuracy, recall, precision, F1, AUC, confusion matrix.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    y_pred_bin = (y_pred_flat >= threshold).astype(int)

    acc = accuracy_score(y_true_flat, y_pred_bin)
    rec = recall_score(y_true_flat, y_pred_bin)
    prec = precision_score(y_true_flat, y_pred_bin)
    f1val = f1_score(y_true_flat, y_pred_bin)

    fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat, pos_label=1)
    auc_roc = auc(fpr, tpr)

    cm = confusion_matrix(y_true_flat, y_pred_bin)

    return {
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1_score": f1val,
        "auc_roc": auc_roc,
        "confusion_matrix": cm
    }

##########################################################################################

def calculate_map_50(y_true, y_pred, threshold=0.5):
    """
    Calculate mean Average Precision at IoU threshold 0.5.

    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array.
    y_pred : array_like
        Predicted probability mask array.
    threshold : float
        IoU threshold for considering a positive detection.

    Returns:
    -------
    float
        mAP@0.5 score
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    iou_score = iou(y_true, y_pred_bin)
    return 1.0 if iou_score >= threshold else 0.0

def accuracy(y_true, y_pred, threshold=0.5):
    """
    Calculate pixel-wise accuracy.

    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array.
    y_pred : array_like
        Predicted probability mask array.
    threshold : float
        Classification threshold.

    Returns:
    -------
    float
        Accuracy score
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_bin.flatten()
    
    correct = np.sum(y_true_flat == y_pred_flat)
    total = len(y_true_flat)
    
    return correct / total

def sensitivity(y_true, y_pred, threshold=0.5, smooth=1e-7):
    """
    Calculate sensitivity (true positive rate).

    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array.
    y_pred : array_like
        Predicted probability mask array.
    threshold : float
        Classification threshold.
    smooth : float
        Smoothing factor to avoid division by zero.

    Returns:
    -------
    float
        Sensitivity score
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_bin.flatten()
    
    true_positives = np.sum(y_true_flat * y_pred_flat)
    actual_positives = np.sum(y_true_flat)
    
    return (true_positives + smooth) / (actual_positives + smooth)

def specificity(y_true, y_pred, threshold=0.5, smooth=1e-7):
    """
    Calculate specificity (true negative rate).

    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array.
    y_pred : array_like
        Predicted probability mask array.
    threshold : float
        Classification threshold.
    smooth : float
        Smoothing factor to avoid division by zero.

    Returns:
    -------
    float
        Specificity score
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_bin.flatten()
    
    true_negatives = np.sum((1 - y_true_flat) * (1 - y_pred_flat))
    actual_negatives = np.sum(1 - y_true_flat)
    
    return (true_negatives + smooth) / (actual_negatives + smooth)
    
##########################################################################################

def mean_edge_error(y_true, y_pred, threshold=0.5):
    """
    Calculate Mean Edge Error between predicted and ground truth masks.

    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array.
    y_pred : array_like
        Predicted probability mask array.
    threshold : float
        Threshold for edge detection.
        
    Returns:
    -------
    float
        Mean Edge Error score
    """
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = y_true.astype(np.uint8)

    true_edges = cv2.Canny(y_true_bin, threshold, threshold)
    pred_edges = cv2.Canny(y_pred_bin, threshold, threshold)

    true_dist = cv2.distanceTransform(255 - true_edges, cv2.DIST_L2, 3)
    pred_dist = cv2.distanceTransform(255 - pred_edges, cv2.DIST_L2, 3)

    true_edge_pixels = true_edges > 0
    pred_edge_pixels = pred_edges > 0

    if np.sum(true_edge_pixels) == 0 or np.sum(pred_edge_pixels) == 0:
        return 0.0

    true_to_pred = np.mean(pred_dist[true_edge_pixels])
    pred_to_true = np.mean(true_dist[pred_edge_pixels])

    return (true_to_pred + pred_to_true) / 2.0

def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error between predicted and ground truth masks.
    
    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array.
    y_pred : array_like
        Predicted probability mask array.
        
    Returns:
    -------
    float
        Mean Absolute Error score
    """
    return np.mean(np.abs(y_true - y_pred))

##########################################################################################

def get_boundary_region(mask, kernel_size=3):
    """Helper function to get boundary region of mask"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    boundary = dilated - eroded
    return boundary

def closed_ended_accuracy(y_true, y_pred, threshold=0.5, kernel_size=3):
    """
    Calculate accuracy for boundary regions (closed-ended)
    """
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    boundary = get_boundary_region(y_true, kernel_size)
    
    correct = np.sum((y_true == y_pred_bin) & (boundary == 1))
    total = np.sum(boundary)
    
    return correct / (total + 1e-7)

def open_ended_accuracy(y_true, y_pred, threshold=0.5, kernel_size=3):
    """
    Calculate accuracy for interior regions (open-ended)
    """
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    boundary = get_boundary_region(y_true, kernel_size)
    
    interior = 1 - boundary
    correct = np.sum((y_true == y_pred_bin) & (interior == 1))
    total = np.sum(interior)
    
    return correct / (total + 1e-7)

def overall_accuracy(y_true, y_pred, threshold=0.5, w1=0.5, w2=0.5):
    """
    Calculate weighted combination of closed and open-ended accuracy
    """
    closed_acc = closed_ended_accuracy(y_true, y_pred, threshold)
    open_acc = open_ended_accuracy(y_true, y_pred, threshold)
    
    return w1 * closed_acc + w2 * open_acc

def region_specific_auc(y_true, y_pred, region_type='all'):
    """
    Calculate AUC score for specific regions
    region_type: 'boundary', 'interior', or 'all'
    """
    if region_type == 'boundary':
        mask = get_boundary_region(y_true)
    elif region_type == 'interior':
        mask = 1 - get_boundary_region(y_true)
    else:
        mask = np.ones_like(y_true)
    
    y_true_masked = y_true[mask == 1]
    y_pred_masked = y_pred[mask == 1]
    
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true_masked, y_pred_masked)
    except:
        return 0.0

##########################################################################################
## need further implementation ##
##########################################################################################

class TRECEvaluator:
    def __init__(self):
        """Initialize TREC evaluator"""
        self.relevance_levels = {
            1: 'relevant',
            0: 'non-relevant'
        }

    def precision_at_k(self, y_true, y_pred, k):
        """
        Calculate precision@k for binary masks
        """
        y_pred_bin = (y_pred >= 0.5).astype(np.float32)
        
        y_pred_flat = y_pred_bin.flatten()
        y_true_flat = y_true.flatten()
        
        if k > len(y_pred_flat):
            k = len(y_pred_flat)
            
        top_k_idx = np.argsort(y_pred_flat)[-k:]
        true_positives = np.sum(y_true_flat[top_k_idx])
        
        return true_positives / k

    def average_precision(self, y_true, y_pred):
        """
        Calculate Average Precision
        """
        y_pred_bin = (y_pred >= 0.5).astype(np.float32)
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred_bin.flatten()
        
        relevant_docs = np.sum(y_true_flat)
        if relevant_docs == 0:
            return 0.0
            
        precisions = []
        running_tp = 0
        
        for i in range(len(y_pred_flat)):
            if y_true_flat[i] == 1:
                running_tp += 1
                precisions.append(running_tp / (i + 1))
                
        return np.mean(precisions) if precisions else 0.0

    def ndcg(self, y_true, y_pred, k=None):
        """
        Calculate Normalized Discounted Cumulative Gain
        """
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()
        
        if k is None:
            k = len(y_pred_flat)
            
        pred_indices = np.argsort(y_pred_flat)[::-1][:k]
        
        dcg = np.sum([y_true_flat[i] / np.log2(idx + 2) 
                      for idx, i in enumerate(pred_indices)])
        
        ideal_indices = np.argsort(y_true_flat)[::-1][:k]
        idcg = np.sum([y_true_flat[i] / np.log2(idx + 2) 
                       for idx, i in enumerate(ideal_indices)])
        
        return dcg / idcg if idcg > 0 else 0.0

##########################################################################################

def ndcg_at_k(y_true, y_pred, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain at k
    
    Parameters:
    ----------
    y_true : array_like
        Ground truth relevance scores
    y_pred : array_like
        Predicted relevance scores
    k : int
        Cutoff for calculation
        
    Returns:
    -------
    float
        NDCG@k score
    """
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    
    if k > len(y_pred_flat):
        k = len(y_pred_flat)
        
    pred_indices = np.argsort(y_pred_flat)[::-1][:k]
    
    dcg = np.sum([y_true_flat[i] / np.log2(idx + 2) 
                  for idx, i in enumerate(pred_indices)])
    
    ideal_indices = np.argsort(y_true_flat)[::-1][:k]
    idcg = np.sum([y_true_flat[i] / np.log2(idx + 2) 
                   for idx, i in enumerate(ideal_indices)])
    
    return dcg / idcg if idcg > 0 else 0.0
    
def ndcd_at_k(y_true, y_pred, k=10):
    """
    Calculate Normalized Discounted Correlation Distance at k
    
    Parameters:
    ----------
    y_true : array_like
        Ground truth relevance scores
    y_pred : array_like
        Predicted relevance scores
    k : int
        Cutoff for calculation
        
    Returns:
    -------
    float
        NDCD@k score
    """
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    
    if k > len(y_pred_flat):
        k = len(y_pred_flat)
        
    pred_indices = np.argsort(y_pred_flat)[::-1][:k]
    ideal_indices = np.argsort(y_true_flat)[::-1][:k]
    
    pred_ranks = np.arange(k) + 1
    true_ranks = np.array([np.where(ideal_indices == i)[0][0] + 1 for i in pred_indices])
    
    dcd = np.sum(np.abs(pred_ranks - true_ranks) / np.log2(pred_ranks + 1))
    
    worst_ranks = np.array(range(k, 0, -1))
    max_dcd = np.sum(np.abs(np.arange(1, k+1) - worst_ranks) / np.log2(np.arange(1, k+1) + 1))
    
    return 1 - (dcd / max_dcd) if max_dcd > 0 else 0.0

##########################################################################################

def bleu_score(y_true, y_pred, n=1):
    """
    Calculate BLEU-n score
    
    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array
    y_pred : array_like
        Predicted binary mask array  
    n : int
        n-gram size (1,2,3)
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    true_seq = ''.join(map(str, y_true_flat.astype(int)))
    pred_seq = ''.join(map(str, y_pred_flat.astype(int)))
    
    true_ngrams = [true_seq[i:i+n] for i in range(len(true_seq)-n+1)]
    pred_ngrams = [pred_seq[i:i+n] for i in range(len(pred_seq)-n+1)]
    
    if not pred_ngrams or not true_ngrams:
        return 0.0
        
    matches = sum(1 for ng in pred_ngrams if ng in true_ngrams)
    
    return matches / len(pred_ngrams)

##########################################################################################

def exact_match(y_true, y_pred, threshold=0.5):
    """
    Calculate Exact Match score
    
    Parameters:
    ----------
    y_true : array_like
        Ground truth binary mask array
    y_pred : array_like  
        Predicted probability mask array
    threshold : float
        Classification threshold
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    return np.array_equal(y_true, y_pred_bin)


##########################################################################################

def meteor_score(y_true, y_pred, alpha=0.9):
    """
    Calculate METEOR score using alignments and stemming
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    matches = np.sum(y_true_flat == y_pred_flat)
    chunks = _get_chunks(y_true_flat, y_pred_flat)
    
    if matches == 0:
        return 0.0
        
    precision = matches / len(y_pred_flat)
    recall = matches / len(y_true_flat)
    
    if precision == 0 or recall == 0:
        return 0.0
        
    fmean = 1.0 / ((1.0 - alpha) / precision + alpha / recall)
    penalty = 0.5 * (chunks / matches)
    
    return fmean * (1.0 - penalty)

def rouge_acc(y_true, y_pred, n=1):
    """
    Calculate ROUGE-ACC score using n-gram matching
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    true_ngrams = _get_ngrams(y_true_flat, n)
    pred_ngrams = _get_ngrams(y_pred_flat, n)
    
    overlap = len(true_ngrams.intersection(pred_ngrams))
    
    if len(true_ngrams) == 0 or len(pred_ngrams) == 0:
        return 0.0
        
    precision = overlap / len(pred_ngrams)
    recall = overlap / len(true_ngrams)
    
    if precision == 0 or recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def factent_score(y_true, y_pred, threshold=0.5):
    """
    Calculate Factual Entailment score
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_bin.flatten()

    entails = np.all(np.logical_or(y_pred_flat == 0, y_true_flat == y_pred_flat))

    return float(entails)

def _get_chunks(true_seq, pred_seq):
    """Helper: count sequence chunks"""
    chunks = 0
    in_chunk = False
    
    for t, p in zip(true_seq, pred_seq):
        if t == p:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False
            
    return chunks

def _get_ngrams(sequence, n):
    """
    Helper: get n-grams from sequence
    """
    return set(tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1))

##########################################################################################

def miou(y_true, y_pred, num_classes=None, smooth=1):
    """
    Calculate Mean Intersection over Union across all classes.

    Parameters:
    ----------
    y_true : array_like
        Ground truth mask array. Shape: (N, H, W) or (H, W)
    y_pred : array_like
        Predicted mask array. Shape: (N, H, W) or (H, W)
    num_classes : int, optional
        Number of classes. If None, derived from data
    smooth : float, optional
        Smoothing factor to avoid division by zero

    Returns:
    -------
    float
        mIoU score averaged across all classes
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if len(y_true.shape) == 2:
        y_true = np.expand_dims(y_true, axis=0)
        y_pred = np.expand_dims(y_pred, axis=0)
        
    if num_classes is None:
        num_classes = len(np.unique(y_true))
    
    ious = []
    for cls in range(num_classes):
        true_mask = (y_true == cls).astype(np.float32)
        pred_mask = (y_pred == cls).astype(np.float32)
        class_iou = iou(true_mask, pred_mask, smooth)
        
        if np.sum(true_mask) > 0:
            ious.append(class_iou)
            
    return np.mean(ious) if ious else 0.0

######################################################################################

def logp_score(y_true, y_pred, threshold=0.5):
    """
    Calculate LogP score (octanol-water partition coefficient)
    """
    try:
        y_pred_bin = (y_pred >= threshold).astype(np.float32)
        mol = array_to_mol(y_pred_bin)
        if mol is None:
            return 0.0
        return MolLogP(mol)
    except:
        return 0.0

def qed_score(y_true, y_pred, threshold=0.5):
    """
    Calculate Quantitative Estimate of Drug-likeness
    """
    try:
        y_pred_bin = (y_pred >= threshold).astype(np.float32)
        mol = array_to_mol(y_pred_bin)
        if mol is None:
            return 0.0
        return QED.default(mol)
    except:
        return 0.0

def sa_score(y_true, y_pred, threshold=0.5):
    """
    Calculate Synthetic Accessibility score
    """
    try:
        y_pred_bin = (y_pred >= threshold).astype(np.float32)
        mol = array_to_mol(y_pred_bin)
        if mol is None:
            return 0.0
        return sascorer.calculateScore(mol)
    except:
        return 0.0

def array_to_mol(array):
    """Helper: Convert binary array to RDKit molecule"""
    try:
        # would need some domain specific logic
        # put a placeholder for now
        return Chem.MolFromSmiles('C')  
    except:
        return None

######################################################################################
