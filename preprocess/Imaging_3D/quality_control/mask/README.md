<div align="center">

# 🏥 Quality-Sentinel:<br>Label Quality Evaluation for Medical Image Segmentation

### *Estimating Label Quality and Errors in Medical Segmentation Datasets*

</div>

---

## 🌟 Highlights

<table>
<tr>
<td width="33%" align="center"><b>🔍 Quality Assessment</b><br>Automatic evaluation of segmentation mask quality</td>
<td width="33%" align="center"><b>📊 142 Organ Support</b><br>Comprehensive coverage of anatomical structures</td>
<td width="33%" align="center"><b>⚡ Efficient Design</b><br>2D model for fast inference with minimal resources</td>
</tr>
</table>

## 📋 Abstract

Quality Sentinel is a label quality evaluation tool for medical image segmentation that helps diagnose data quality in large-scale CT image segmentation datasets. The model takes an image-label pair as input and estimates the Dice Similarity Coefficient (DSC) of the mask compared to ground truth.

## 🚀 Getting Started

### 1. Installation

Clone this repository and navigate to the folder:

```bash
git clone https://github.com/yourusername/Quality-Sentinel.git
cd Quality-Sentinel
```

### 2. Install Package

```bash
# Create a new conda environment
conda create -n quality_sentinel python=3.9 -y
conda activate quality_sentinel

# Install PyTorch (adjust cuda version as needed)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 3. Download Resources

#### 📦 Pre-trained Model & Data
Download from [Google Drive](https://drive.google.com/file/d/10K_D67vXIG8w41hTIhFXQjLRcXgExCRA/view?usp=sharing):
- `best_resnet50_model_40_samples.pth` - Pre-trained Quality Sentinel model

Place it directly in the project root

### 4. Data Format

We take `LiTS` dataset as example, which is available at [URL](https://competitions.codalab.org/competitions/17094).

### LiTSMaskDataset Support

The `LiTSMaskDataset` class supports medical imaging data in the following format:

#### **Expected Directory Structure:**
```
your_dataset/
├── volumns/           # CT volumes
│   ├── volume-0.nii.gz
│   ├── volume-1.nii.gz
│   └── ...
└── segmentations/     # Corresponding masks
    ├── segmentation-0.nii.gz
    ├── segmentation-1.nii.gz
    └── ...
```

#### **Data Format Requirements:**
- **CT Images**: NIfTI format (`.nii` or `.nii.gz`)
  - 3D volumes with shape `[D, H, W]` (depth, height, width)
  - Original HU (Hounsfield Unit) values
  - File naming: `volume-{id}.nii.gz`

- **Segmentation Masks**: NIfTI format (`.nii` or `.nii.gz`)
  - Same dimensions as corresponding CT volume
  - Binary or multi-class integer labels
  - For LiTS dataset: values 1 and 2 are treated as foreground (liver)
  - File naming: `segmentation-{id}.nii.gz`

### 5. Class Index Mapping

The `class_idx` parameter corresponds to organ classes defined in `DAP_Atlas_label_name.csv`. Here are some key mappings:

| class_idx | Organ Name | Common Usage |
|:---------:|:-----------|:-------------|
| 0 | Background | Background pixels |
| 6 | Esophagus | GI tract segmentation |
| 7 | Stomach | GI tract segmentation |
| 10 | Colon | GI tract segmentation |
| **13** | **Liver** | **Our example in demos** |
| 14 | Pancreas | Abdominal organs |
| 15/16 | Kidney (L/R) | Renal segmentation |
| 26 | Spleen | Abdominal organs |

**Full mapping available in:** [`DAP_Atlas_label_name.csv`](./DAP_Atlas_label_name.csv)

### 6. Quick Start

**🚀 Test on LiTS Dataset**

To quickly test Quality Sentinel on the LiTS dataset:

```bash
# Run evaluation on LiTS liver segmentation
python test_lits.py
```

This script will:
1. Load the pre-trained Quality Sentinel model
2. Process LiTS liver segmentation masks (class_idx=13)
3. Extract middle slices from each volume
4. Predict quality scores (DSC) for each mask
5. Output average predicted Dice score and detailed results

**Expected Output:**
```
Loading LiTS data...
LiTS Data Amount: 131
100%|████████████| 131/131 [02:45<00:00, 1.26s/it]

Results Summary:
Average predicted Dice: 0.7153
```

**💡 Custom Dataset Testing**

You can modify `test_lits.py` to test on your own dataset:

```python
# Change the mask directory path
mask_dir = "/path/to/your/segmentations"

# Change the class index for different organs
lits_dataset = LiTSMaskDataset(
    mask_dir=mask_dir,
    transform_ct=transform_ct,
    transform_mask=transform_mask,
    mode='test',
    num_samples=10  # Optional: limit samples for quick testing
)

# Update the class index in the inference loop
mask_class[0].item()  # Current: 13 (liver)
```

## 🖥️ System Requirements

- **Hardware**: Tested on 1× NVIDIA A5000 GPU
- **OS**: Ubuntu 20.04
- **Memory**: 16GB RAM minimum
- **Storage**: ~2GB for model and sample data

## 📚 Citation

If you find Quality Sentinel useful in your research, please consider citing:

```bibtex
@article{chen2024quality,
title={Quality Sentinel: Estimating Label Quality and Errors in Medical Segmentation Datasets},
author={Chen, Yixiong and Zhou, Zongwei and Yuille, Alan},
journal={arXiv preprint arXiv:2406.00327},
year={2024}
}
```

## 🙏 Acknowledgements

We gratefully acknowledge:
- **[Quality-Sentinel](https://github.com/Schuture/Quality-Sentinel)**

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors through the paper.













