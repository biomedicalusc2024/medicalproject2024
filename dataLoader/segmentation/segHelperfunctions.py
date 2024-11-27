import numpy as np

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
