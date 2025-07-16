<div align="center">

# 🧠 TorchIO:<br>A Python Library for Efficient Medical Image I/O, Preprocessing, Augmentation and Patch-based Sampling in Deep Learning

### *Accelerating 3D Medical Image Processing for Deep Learning Applications*

</div>

---

## 📋 Abstract

TorchIO is a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of 3D medical images in deep learning, following PyTorch conventions. It includes multiple processing and augmentation tools for scalar and spatial transforms, supports most medical image formats.

## 🚀 Getting Started

### 1. Installation

#### Basic Installation

Install the latest PyTorch version using light-the-torch (recommended):

```bash
pip install light-the-torch && ltt install torch
```

Install TorchIO:

```bash
pip install torchio
```

#### Optional Features

For visualization capabilities:

```bash
pip install torchio[plot]
```

### 2. Upgrade

To upgrade to the latest version:

```bash
pip install --upgrade torchio
```

### 3. Development Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/fepegar/torchio.git
cd torchio
pip install -e .
```

### 4. Quick Start

**🚀 Basic Usage Example**

```python
import torch
import torchio as tio

# Each instance of tio.Subject is passed arbitrary keyword arguments.
# Typically, these arguments will be instances of tio.Image
subject_a = tio.Subject(
    t1=tio.ScalarImage('subject_a.nii.gz'),
    label=tio.LabelMap('subject_a.nii'),
    diagnosis='positive',
)

# Image files can be in any format supported by SimpleITK or NiBabel, including DICOM
subject_b = tio.Subject(
    t1=tio.ScalarImage('subject_b_dicom_folder'),
    label=tio.LabelMap('subject_b_seg.nrrd'),
    diagnosis='negative',
)

# Images may also be created using PyTorch tensors or NumPy arrays
tensor_4d = torch.rand(4, 100, 100, 100)
subject_c = tio.Subject(
    t1=tio.ScalarImage(tensor=tensor_4d),
    label=tio.LabelMap(tensor=(tensor_4d > 0.5)),
    diagnosis='negative',
)

subjects_list = [subject_a, subject_b, subject_c]

# Let's use one preprocessing transform and one augmentation transform
# This transform will be applied only to scalar images:
rescale = tio.RescaleIntensity(out_min_max=(0, 1))

# As RandomAffine is faster then RandomElasticDeformation, we choose to
# apply RandomAffine 80% of the times and RandomElasticDeformation the rest
# Also, there is a 25% chance that none of them will be applied
spatial = tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    },
    p=0.75,
)

# Transforms can be composed as in torchvision.transforms
transforms = [rescale, spatial]
transform = tio.Compose(transforms)

# SubjectsDataset is a subclass of torch.data.utils.Dataset
subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)

# Images are processed in parallel thanks to a SubjectsLoader
# (which inherits from torch.utils.data.DataLoader)
training_loader = tio.SubjectsLoader(
    subjects_dataset,
    batch_size=4,
    num_workers=4,
    shuffle=True,
)

# Training epoch
for subjects_batch in training_loader:
    inputs = subjects_batch['t1'][tio.DATA]
    target = subjects_batch['label'][tio.DATA]
```

**🧩 Patch-based Training**

```python
# Define patch sampler
sampler = tio.UniformSampler(patch_size=96)

# Create queue for patch-based training
patches_queue = tio.Queue(
    subjects_dataset,
    max_length=300,
    samples_per_volume=10,
    sampler=sampler,
    num_workers=4,
)

# Training with patches
for patches_batch in patches_queue:
    inputs = patches_batch['t1'][tio.DATA]
    targets = patches_batch['label'][tio.DATA]
    # Train on patches...
```

### Supported Formats

TorchIO supports all formats readable by SimpleITK and NiBabel:
- NIfTI (`.nii`, `.nii.gz`)
- DICOM series
- NRRD (`.nrrd`)
- MetaImage (`.mha`, `.mhd`)
- Analyze (`.hdr`, `.img`)
- And many more...

## 🎓 Tutorials

### Interactive Notebooks

Access the Google Colab tutorials:
- [Getting Started with TorchIO](https://colab.research.google.com/github/TorchIO-project/torchio-notebooks/blob/main/notebooks/TorchIO_tutorial.ipynb)
- [Data Augmentation](https://colab.research.google.com/github/TorchIO-project/torchio-notebooks/blob/main/notebooks/Data_preprocessing_and_augmentation_using_TorchIO_a_tutorial.ipynb)
- [Inference](https://colab.research.google.com/github/TorchIO-project/torchio-notebooks/blob/main/notebooks/Brain_parcellation_with_TorchIO_and_HighRes3DNet.ipynb)
- [TorchIO + MONAI](https://colab.research.google.com/github/TorchIO-project/torchio-notebooks/blob/main/notebooks/TorchIO_MONAI_PyTorch_Lightning.ipynb)


## 🖥️ System Requirements

- **Python**: 3.7+
- **PyTorch**: 1.7+
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **GPU**: CUDA-capable GPU for training (optional)

## 📚 Citation

If you use TorchIO in your research, please cite:

```bibtex
@article{perez-garcia_torchio_2021,
    title = {{TorchIO}: a {Python} library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning},
    journal = {Computer Methods and Programs in Biomedicine},
    pages = {106236},
    year = {2021},
    issn = {0169-2607},
    doi = {https://doi.org/10.1016/j.cmpb.2021.106236},
    url = {https://www.sciencedirect.com/science/article/pii/S0169260721003102},
    author = {P{\'e}rez-Garc{\'i}a, Fernando and Sparks, Rachel and Ourselin, S{\'e}bastien},
}
```

## 🙏 Acknowledgements

We gratefully acknowledge:
- **[TorchIO](https://github.com/TorchIO-project/torchio)**

