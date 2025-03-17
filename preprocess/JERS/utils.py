import os
import numpy as np
import SimpleITK as sitk


def norm_to_0_1(array):
    min_val = np.min(array)
    max_val = np.max(array)
    
    if np.isclose(max_val, min_val):
        normalized_array = np.zeros_like(array, dtype=np.float32)
    else:
        normalized_array = (array - min_val) / (max_val - min_val)
    
    return normalized_array


def image_to_square_v2(img,size):
    img_new=np.zeros((size,size,size))
    z,x,y=img.shape
    if z<= size:
        az = (size-z)//2
        img_new_start_z=az
        img_new_end_z=az+z
        img_start_z=0
        img_end_z=z

    elif z> size:
        az = (z-size)//2
        img_new_start_z=0
        img_new_end_z=size
        img_start_z=az
        img_end_z=az+size

    if x<= size:
        ax = (size-x)//2
        img_new_start_x=ax
        img_new_end_x=ax+x
        img_start_x=0
        img_end_x=x

    elif x> size:
        ax = (x-size)//2
        img_new_start_x=0
        img_new_end_x=size
        img_start_x=ax
        img_end_x=ax+size

    if y<= size:
        ay = (size-y)//2
        img_new_start_y=ay
        img_new_end_y=ay+y
        img_start_y=0
        img_end_y=y

    elif y> size:
        ay = (y-size)//2
        img_new_start_y=0
        img_new_end_y=size
        img_start_y=ay
        img_end_y=ay+size

    img_new[img_new_start_z:img_new_end_z,img_new_start_x:img_new_end_x,img_new_start_y:img_new_end_y] = img[img_start_z:img_end_z,img_start_x:img_end_x,img_start_y:img_end_y]   
    return img_new


def save_nii_any(image, path, index):
    ref_img_GetOrigin = (0.0, 0.0, 0.0)
    ref_img_GetDirection = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    ref_img_GetSpacing = (1.0, 1.0, 1.0)

    img = sitk.GetImageFromArray(image.squeeze().cpu().detach().numpy())

    img.SetOrigin(ref_img_GetOrigin)
    img.SetDirection(ref_img_GetDirection)
    img.SetSpacing(ref_img_GetSpacing)

    image_path = os.path.join(path, f"{index}.nii.gz")
    os.makedirs(path, exist_ok=True)
    sitk.WriteImage(img, image_path)

    return image_path