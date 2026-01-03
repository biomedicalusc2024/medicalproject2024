# Brain Image Preprocessing Tutorial (Using DeepPrep)

## What is DeepPrep?

DeepPrep is a deep learning-powered neuroimaging preprocessing pipeline (Nature Methods, 2025) that processes both **structural (T1)** and **functional (fMRI)** brain scans.

**Key advantages:**
- 10× faster than fMRIPrep (~27 min vs 4-8 hours per subject)
- 100% completion rate vs 85% for fMRIPrep
- Validated on 55,000+ scans

---

## Installation

### Docker (Recommended)

```bash
# Pull latest image
docker pull pbfslab/deepprep:25.1.0

# Verify installation
docker run --rm pbfslab/deepprep:25.1.0 --version
```

### Singularity (HPC clusters)

```bash
singularity build deepprep-25.1.0.sif docker://pbfslab/deepprep:25.1.0
```

**Requirements:** 16GB RAM, 5-10GB storage per subject

---

## Data Preparation

### BIDS Format Required

DeepPrep only accepts BIDS-formatted data:

```
my_dataset/
├── sub-001/
│   ├── anat/
│   │   └── sub-001_T1w.nii.gz
│   └── func/
│       ├── sub-001_task-rest_bold.nii.gz
│       └── sub-001_task-rest_bold.json
├── sub-002/
│   └── ...
└── dataset_description.json
```

### Convert DICOM to BIDS

```bash
# Install dcm2bids
pip install dcm2bids

# Convert
dcm2bids -d dicom_folder -p 001 -c config.json -o bids_output

# Validate
pip install bids-validator
bids-validator bids_output
```

### Quick BIDS Creation Script

```python
import os
import shutil
from pathlib import Path

def create_bids_structure(base_dir, subject_id, t1_file=None, fmri_file=None):
    """
    Create BIDS directory structure for a subject

    Args:
        base_dir: Root BIDS directory
        subject_id: Subject ID (e.g., '001')
        t1_file: Path to T1 NIfTI file
        fmri_file: Path to fMRI NIfTI file
    """
    sub_dir = Path(base_dir) / f'sub-{subject_id}'
    anat_dir = sub_dir / 'anat'
    func_dir = sub_dir / 'func'

    anat_dir.mkdir(parents=True, exist_ok=True)
    func_dir.mkdir(parents=True, exist_ok=True)

    # Copy T1
    if t1_file:
        dest = anat_dir / f'sub-{subject_id}_T1w.nii.gz'
        shutil.copy(t1_file, dest)
        print(f"Copied T1: {dest}")

    # Copy fMRI
    if fmri_file:
        dest = func_dir / f'sub-{subject_id}_task-rest_bold.nii.gz'
        shutil.copy(fmri_file, dest)
        print(f"Copied fMRI: {dest}")

    # Create dataset_description.json
    desc_file = Path(base_dir) / 'dataset_description.json'
    if not desc_file.exists():
        import json
        desc = {
            "Name": "My Dataset",
            "BIDSVersion": "1.6.0"
        }
        with open(desc_file, 'w') as f:
            json.dump(desc, f, indent=2)

# Example usage
create_bids_structure(
    base_dir='/data/bids',
    subject_id='001',
    t1_file='/raw/subject1_t1.nii.gz',
    fmri_file='/raw/subject1_fmri.nii.gz'
)
```

---

## Running DeepPrep

### Basic Command

```bash
docker run --rm -it \
    -v /path/to/bids_data:/input:ro \
    -v /path/to/output:/output \
    -v /path/to/freesurfer_license:/license:ro \
    pbfslab/deepprep:25.1.0 \
    /input /output participant \
    --participant-label 001 002 003 \
    --fs-license-file /license/license.txt
```

### Common Options

```bash
# T1 only (no fMRI)
--anat-only

# Skip distortion correction (faster)
--ignore fieldmaps

# Use GPU
--device gpu

# Parallel processing
--nthreads 8

# Specific output spaces
--output-spaces MNI152NLin2009cAsym fsaverage6

# Lower memory usage
--low-mem
```

### Batch Processing Script

```bash
#!/bin/bash
# process_all.sh

BIDS_DIR="/data/bids"
OUTPUT_DIR="/data/output"
LICENSE="/license/license.txt"

# Process subjects in parallel
for subj in 001 002 003 004 005; do
    docker run --rm -d \
        -v ${BIDS_DIR}:/input:ro \
        -v ${OUTPUT_DIR}:/output \
        -v ${LICENSE}:/license:ro \
        pbfslab/deepprep:25.1.0 \
        /input /output participant \
        --participant-label ${subj} \
        --fs-license-file /license \
        --nthreads 4 &
done

wait
echo "All subjects processed"
```

### HPC Cluster (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=deepprep
#SBATCH --array=1-100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Load singularity
module load singularity

# Get subject ID from array
SUBJECTS=(001 002 003 ...)  # List all subjects
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID-1]}

# Run DeepPrep
singularity run \
    --bind /data/bids:/input \
    --bind /data/output:/output \
    deepprep-25.1.0.sif \
    /input /output participant \
    --participant-label ${SUBJ} \
    --nthreads 8
```

---

## Output Structure

```
output/
├── deepprep/
│   ├── sub-001/
│   │   ├── anat/
│   │   │   ├── sub-001_space-MNI152_T1w.nii.gz        # Normalized T1
│   │   │   ├── sub-001_label-GM_probseg.nii.gz        # Gray matter
│   │   │   ├── sub-001_label-WM_probseg.nii.gz        # White matter
│   │   │   └── sub-001_label-CSF_probseg.nii.gz       # CSF
│   │   ├── func/
│   │   │   ├── sub-001_task-rest_space-MNI152_bold.nii.gz  # Normalized BOLD
│   │   │   └── sub-001_task-rest_confounds.tsv             # Confounds
│   │   └── surf/                                       # Surface data
│   └── sub-001.html                                    # QC report
└── work/                                               # Intermediate files
```

---

## Loading Data in Python

### Single Subject

```python
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

class DeepPrepLoader:
    def __init__(self, output_dir, subject_id):
        self.output_dir = Path(output_dir) / 'deepprep' / f'sub-{subject_id}'
        self.subject_id = subject_id

    def load_t1(self, space='MNI152'):
        """Load preprocessed T1 image"""
        path = self.output_dir / 'anat' / f'sub-{self.subject_id}_space-{space}_T1w.nii.gz'
        img = nib.load(path)
        return img.get_fdata(), img.affine

    def load_bold(self, task='rest', space='MNI152'):
        """Load preprocessed BOLD image"""
        path = self.output_dir / 'func' / f'sub-{self.subject_id}_task-{task}_space-{space}_bold.nii.gz'
        img = nib.load(path)
        return img.get_fdata(), img.affine

    def load_confounds(self, task='rest'):
        """Load confounds for denoising"""
        path = self.output_dir / 'func' / f'sub-{self.subject_id}_task-{task}_confounds.tsv'
        return pd.read_csv(path, sep='\t')

    def load_tissue_masks(self):
        """Load GM, WM, CSF probability maps"""
        masks = {}
        for tissue in ['GM', 'WM', 'CSF']:
            path = self.output_dir / 'anat' / f'sub-{self.subject_id}_label-{tissue}_probseg.nii.gz'
            masks[tissue] = nib.load(path).get_fdata()
        return masks

# Usage
loader = DeepPrepLoader('/data/output', '001')
t1_data, t1_affine = loader.load_t1()
bold_data, bold_affine = loader.load_bold()
confounds = loader.load_confounds()

print(f"T1 shape: {t1_data.shape}")        # (182, 218, 182)
print(f"BOLD shape: {bold_data.shape}")    # (182, 218, 182, 200)
print(f"Confounds: {confounds.columns.tolist()}")
```

### Batch Loading

```python
import glob
from concurrent.futures import ThreadPoolExecutor

def load_all_subjects(output_dir, n_workers=4):
    """Load all preprocessed subjects in parallel"""
    base_path = Path(output_dir) / 'deepprep'
    subject_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    subject_ids = [d.name.replace('sub-', '') for d in subject_dirs]

    def load_subject(subj_id):
        try:
            loader = DeepPrepLoader(output_dir, subj_id)
            t1_data, _ = loader.load_t1()
            return {
                'subject_id': subj_id,
                'data': t1_data,
                'success': True
            }
        except Exception as e:
            print(f"Failed to load {subj_id}: {e}")
            return {'subject_id': subj_id, 'success': False}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(load_subject, subject_ids))

    successful = [r for r in results if r['success']]
    print(f"Loaded {len(successful)}/{len(subject_ids)} subjects")
    return successful

# Usage
dataset = load_all_subjects('/data/output')
```

---

## Quality Control

### Extract QC Metrics

```python
def extract_qc_metrics(output_dir, subject_id, task='rest'):
    """Extract key QC metrics from confounds"""
    loader = DeepPrepLoader(output_dir, subject_id)
    confounds = loader.load_confounds(task)

    # Framewise displacement (head motion)
    fd = confounds['framewise_displacement'].values
    fd_mean = np.nanmean(fd)
    fd_max = np.nanmax(fd)

    # DVARS (signal change rate)
    if 'dvars' in confounds.columns:
        dvars = confounds['dvars'].values
        dvars_mean = np.nanmean(dvars)
    else:
        dvars_mean = None

    # Count high-motion timepoints
    high_motion_tps = np.sum(fd > 0.5)

    return {
        'subject_id': subject_id,
        'fd_mean': fd_mean,
        'fd_max': fd_max,
        'dvars_mean': dvars_mean,
        'high_motion_tps': high_motion_tps,
        'total_tps': len(fd)
    }

# Batch QC
subjects = ['001', '002', '003']
qc_results = [extract_qc_metrics('/data/output', s) for s in subjects]
qc_df = pd.DataFrame(qc_results)

# Filter by quality
good_subjects = qc_df[qc_df['fd_mean'] < 0.5]['subject_id'].tolist()
print(f"Subjects passing QC: {len(good_subjects)}/{len(subjects)}")
```

### Automated QC Report

```python
import matplotlib.pyplot as plt

def plot_qc_summary(qc_df, output_file='qc_summary.png'):
    """Generate QC summary plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # FD distribution
    axes[0, 0].hist(qc_df['fd_mean'], bins=20, edgecolor='black')
    axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
    axes[0, 0].set_xlabel('Mean FD (mm)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Head Motion Distribution')
    axes[0, 0].legend()

    # DVARS distribution
    axes[0, 1].hist(qc_df['dvars_mean'].dropna(), bins=20, edgecolor='black')
    axes[0, 1].set_xlabel('Mean DVARS')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Signal Change Distribution')

    # High-motion timepoints
    axes[1, 0].bar(range(len(qc_df)), qc_df['high_motion_tps'])
    axes[1, 0].set_xlabel('Subject Index')
    axes[1, 0].set_ylabel('High-Motion TPs')
    axes[1, 0].set_title('High-Motion Timepoints per Subject')

    # Pass/fail summary
    passed = (qc_df['fd_mean'] < 0.5).sum()
    failed = len(qc_df) - passed
    axes[1, 1].bar(['Passed', 'Failed'], [passed, failed])
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('QC Summary')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"QC report saved to {output_file}")

# Usage
plot_qc_summary(qc_df)
```

---

## Denoising fMRI Data

### Confound Regression

```python
from sklearn.linear_model import LinearRegression

def denoise_bold(bold_data, confounds, confound_columns):
    """
    Remove confounds from BOLD data

    Args:
        bold_data: 4D array (x, y, z, time)
        confounds: DataFrame with confound timeseries
        confound_columns: List of confound names to regress out

    Returns:
        denoised_data: 4D array with confounds removed
    """
    # Reshape to 2D (voxels × time)
    original_shape = bold_data.shape
    bold_2d = bold_data.reshape(-1, original_shape[-1]).T  # (time, voxels)

    # Select confounds
    X = confounds[confound_columns].fillna(0).values

    # Fit and remove confounds
    lr = LinearRegression()
    denoised = np.zeros_like(bold_2d)

    for i in range(bold_2d.shape[1]):
        y = bold_2d[:, i]
        lr.fit(X, y)
        residuals = y - lr.predict(X)
        denoised[:, i] = residuals

    # Reshape back
    denoised = denoised.T.reshape(original_shape)
    return denoised

# Example usage
loader = DeepPrepLoader('/data/output', '001')
bold_data, affine = loader.load_bold()
confounds = loader.load_confounds()

# Select confounds to regress
confound_cols = [
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'csf', 'white_matter'
]

# Denoise
clean_bold = denoise_bold(bold_data, confounds, confound_cols)

# Save denoised data
clean_img = nib.Nifti1Image(clean_bold, affine)
nib.save(clean_img, 'sub-001_task-rest_denoised.nii.gz')
```

### Temporal Filtering

```python
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=0.01, highcut=0.1, fs=0.5):
    """
    Apply bandpass filter to BOLD timeseries

    Args:
        data: 4D BOLD array
        lowcut: Low frequency cutoff (Hz)
        highcut: High frequency cutoff (Hz)
        fs: Sampling frequency (Hz, typically 1/TR)
    """
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(2, [low, high], btype='band')

    # Filter each voxel's timeseries
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                ts = data[i, j, k, :]
                if np.std(ts) > 0:  # Skip zero voxels
                    filtered[i, j, k, :] = filtfilt(b, a, ts)

    return filtered

# Usage (TR = 2 seconds, fs = 0.5 Hz)
filtered_bold = bandpass_filter(clean_bold, lowcut=0.01, highcut=0.1, fs=0.5)
```

---

## Deep Learning Integration

### PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

class BrainMRIDataset(Dataset):
    def __init__(self, output_dir, subject_ids, modality='T1', space='MNI152', transform=None):
        """
        Args:
            output_dir: DeepPrep output directory
            subject_ids: List of subject IDs
            modality: 'T1' or 'BOLD'
            space: 'MNI152' or other output space
            transform: Optional transform function
        """
        self.output_dir = Path(output_dir)
        self.subject_ids = subject_ids
        self.modality = modality
        self.space = space
        self.transform = transform

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subj_id = self.subject_ids[idx]
        loader = DeepPrepLoader(self.output_dir, subj_id)

        if self.modality == 'T1':
            data, _ = loader.load_t1(space=self.space)
        elif self.modality == 'BOLD':
            data, _ = loader.load_bold(space=self.space)
            # Average over time for static analysis
            data = np.mean(data, axis=-1)

        # Convert to tensor
        data = torch.FloatTensor(data).unsqueeze(0)  # Add channel dimension

        # Normalize
        data = (data - data.mean()) / (data.std() + 1e-8)

        if self.transform:
            data = self.transform(data)

        return data, subj_id

# Usage
subjects = ['001', '002', '003', '004', '005']
dataset = BrainMRIDataset('/data/output', subjects, modality='T1')
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

for batch_data, batch_ids in loader:
    print(f"Batch shape: {batch_data.shape}")  # (2, 1, 182, 218, 182)
    print(f"Batch IDs: {batch_ids}")
    break
```

### 3D CNN Example

```python
import torch.nn as nn

class Brain3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)

        # Calculate flattened size after pooling
        # Input: (182, 218, 182) -> after 3 pools: (22, 27, 22)
        self.fc1 = nn.Linear(128 * 22 * 27 * 22, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Training loop
model = Brain3DCNN(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy labels for demonstration
labels = torch.LongTensor([0, 1])  # Binary classification

for epoch in range(10):
    for batch_data, batch_ids in loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, labels[:len(batch_data)])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### Using Pre-trained Models

```python
from monai.networks.nets import DenseNet121

def create_brain_classifier(pretrained=True):
    """Create 3D DenseNet for brain MRI classification"""
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=2
    )
    return model

# Load with MONAI
import monai.transforms as mt

transforms = mt.Compose([
    mt.ScaleIntensity(),
    mt.RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    mt.RandFlip(prob=0.5, spatial_axis=0)
])

dataset = BrainMRIDataset('/data/output', subjects, transform=transforms)
```

---

## ROI Analysis

### Extract ROI Timeseries

```python
def extract_roi_timeseries(bold_data, atlas_data, roi_labels):
    """
    Extract mean timeseries for each ROI

    Args:
        bold_data: 4D BOLD array (x, y, z, time)
        atlas_data: 3D atlas with integer labels
        roi_labels: List of ROI labels to extract

    Returns:
        timeseries: Array of shape (n_rois, n_timepoints)
    """
    n_timepoints = bold_data.shape[-1]
    timeseries = np.zeros((len(roi_labels), n_timepoints))

    for i, label in enumerate(roi_labels):
        mask = atlas_data == label
        roi_ts = bold_data[mask].mean(axis=0)
        timeseries[i, :] = roi_ts

    return timeseries

# Load atlas (e.g., AAL or Schaefer)
atlas_img = nib.load('atlas_MNI152.nii.gz')
atlas_data = atlas_img.get_fdata()

# Extract timeseries for ROIs 1-10
roi_ts = extract_roi_timeseries(clean_bold, atlas_data, range(1, 11))
print(f"ROI timeseries shape: {roi_ts.shape}")  # (10, 200)
```

### Functional Connectivity

```python
from sklearn.covariance import LedoitWolf

def compute_connectivity(timeseries, method='correlation'):
    """
    Compute functional connectivity matrix

    Args:
        timeseries: Array of shape (n_rois, n_timepoints)
        method: 'correlation' or 'partial_correlation'

    Returns:
        connectivity: Correlation matrix (n_rois, n_rois)
    """
    if method == 'correlation':
        connectivity = np.corrcoef(timeseries)

    elif method == 'partial_correlation':
        # Estimate precision matrix
        estimator = LedoitWolf()
        estimator.fit(timeseries.T)
        precision = estimator.precision_

        # Convert to partial correlation
        d = np.sqrt(np.diag(precision))
        connectivity = -precision / np.outer(d, d)
        np.fill_diagonal(connectivity, 1)

    return connectivity

# Compute connectivity
conn_matrix = compute_connectivity(roi_ts, method='correlation')

# Visualize
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(conn_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Functional Connectivity Matrix')
plt.savefig('connectivity_matrix.png', dpi=300)
```

---

## Utility Functions

### Complete Pipeline Wrapper

```python
class DeepPrepPipeline:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def get_all_subjects(self):
        """Get list of all processed subjects"""
        base = self.output_dir / 'deepprep'
        subjects = [d.name.replace('sub-', '') for d in base.iterdir()
                   if d.is_dir() and d.name.startswith('sub-')]
        return sorted(subjects)

    def quality_control(self, fd_threshold=0.5):
        """Run QC on all subjects and return passing subjects"""
        subjects = self.get_all_subjects()
        qc_results = []

        for subj in subjects:
            try:
                metrics = extract_qc_metrics(self.output_dir, subj)
                qc_results.append(metrics)
            except Exception as e:
                print(f"QC failed for {subj}: {e}")

        qc_df = pd.DataFrame(qc_results)
        passing = qc_df[qc_df['fd_mean'] < fd_threshold]['subject_id'].tolist()

        print(f"QC Summary: {len(passing)}/{len(subjects)} subjects passed")
        return passing, qc_df

    def create_dataset(self, subject_ids, modality='T1'):
        """Create PyTorch dataset for given subjects"""
        return BrainMRIDataset(self.output_dir, subject_ids, modality=modality)

    def batch_denoise(self, subject_ids, confound_cols, output_suffix='_denoised'):
        """Denoise all subjects and save"""
        for subj in subject_ids:
            try:
                loader = DeepPrepLoader(self.output_dir, subj)
                bold, affine = loader.load_bold()
                confounds = loader.load_confounds()

                clean = denoise_bold(bold, confounds, confound_cols)

                out_path = self.output_dir / 'deepprep' / f'sub-{subj}' / 'func' / \
                           f'sub-{subj}_task-rest{output_suffix}.nii.gz'
                nib.save(nib.Nifti1Image(clean, affine), out_path)
                print(f"Denoised: {subj}")

            except Exception as e:
                print(f"Failed to denoise {subj}: {e}")

# Usage
pipeline = DeepPrepPipeline('/data/output')
passing_subjects, qc_df = pipeline.quality_control(fd_threshold=0.5)
dataset = pipeline.create_dataset(passing_subjects, modality='T1')

# Denoise all passing subjects
confounds_to_remove = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
pipeline.batch_denoise(passing_subjects, confounds_to_remove)
```

### Data Augmentation for 3D MRI

```python
import scipy.ndimage as ndi

def augment_3d_mri(data, rotation_range=10, shift_range=5, zoom_range=0.1):
    """
    Apply random augmentation to 3D MRI data

    Args:
        data: 3D array
        rotation_range: Max rotation angle (degrees)
        shift_range: Max shift (voxels)
        zoom_range: Max zoom factor
    """
    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    data = ndi.rotate(data, angle, axes=(0, 1), reshape=False, order=1)

    # Random shift
    shift = [np.random.randint(-shift_range, shift_range) for _ in range(3)]
    data = ndi.shift(data, shift, order=1)

    # Random zoom
    zoom = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    data = ndi.zoom(data, zoom, order=1)

    return data
```

---

## Troubleshooting

### Check Processing Status

```python
def check_processing_status(output_dir):
    """Check which subjects completed successfully"""
    base = Path(output_dir) / 'deepprep'
    subjects = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith('sub-')]

    status = []
    for subj in subjects:
        subj_dir = base / subj

        # Check for key output files
        t1_exists = (subj_dir / 'anat' / f'{subj}_space-MNI152_T1w.nii.gz').exists()
        html_exists = (base / f'{subj}.html').exists()

        status.append({
            'subject': subj,
            't1_processed': t1_exists,
            'qc_report': html_exists,
            'complete': t1_exists and html_exists
        })

    status_df = pd.DataFrame(status)
    print(f"Complete: {status_df['complete'].sum()}/{len(status_df)}")
    return status_df

# Usage
status = check_processing_status('/data/output')
incomplete = status[~status['complete']]['subject'].tolist()
print(f"Incomplete subjects: {incomplete}")
```

### Memory-Efficient Loading

```python
def load_mri_memory_efficient(file_path, downsample_factor=2):
    """Load MRI with downsampling to reduce memory"""
    img = nib.load(file_path)
    data = img.get_fdata()

    # Downsample
    if downsample_factor > 1:
        data = data[::downsample_factor, ::downsample_factor, ::downsample_factor]

    return data

# Load with reduced resolution
t1_lowres = load_mri_memory_efficient('sub-001_T1w.nii.gz', downsample_factor=2)
```

---

## Resources

**Official Documentation:** https://deepprep.readthedocs.io
**GitHub:** https://github.com/pBFSLab/DeepPrep
**Paper:** Nature Methods (2025) - https://www.nature.com/articles/s41592-025-02599-1
**FreeSurfer License:** https://surfer.nmr.mgh.harvard.edu/registration.html (free)

**Learning Resources:**
- BIDS Specification: https://bids-specification.readthedocs.io
- Nilearn (Python neuroimaging): https://nilearn.github.io
- MONAI (Medical imaging deep learning): https://monai.io

---

## Python API Wrapper (Command-line Automation)

**Note:** DeepPrep doesn't provide a native Python API. Below are wrapper functions to call DeepPrep from Python.

### Basic Wrapper Function

```python
import subprocess
from pathlib import Path
from typing import List, Optional

class DeepPrepRunner:
    def __init__(self,
                 docker_image='pbfslab/deepprep:25.1.0',
                 fs_license_path='/path/to/license.txt'):
        """
        Initialize DeepPrep runner

        Args:
            docker_image: DeepPrep Docker image tag
            fs_license_path: Path to FreeSurfer license file
        """
        self.docker_image = docker_image
        self.fs_license = Path(fs_license_path)

        if not self.fs_license.exists():
            raise FileNotFoundError(f"FreeSurfer license not found: {fs_license_path}")

    def run(self,
            bids_dir: str,
            output_dir: str,
            participant_labels: Optional[List[str]] = None,
            anat_only: bool = False,
            nthreads: int = 4,
            device: str = 'cpu',
            skip_bids_validation: bool = False) -> subprocess.CompletedProcess:
        """
        Run DeepPrep preprocessing

        Args:
            bids_dir: BIDS input directory
            output_dir: Output directory
            participant_labels: List of subject IDs (e.g., ['001', '002'])
            anat_only: Only process T1 (skip fMRI)
            nthreads: Number of threads
            device: 'cpu' or 'gpu'
            skip_bids_validation: Skip BIDS validation

        Returns:
            subprocess.CompletedProcess object
        """
        bids_dir = Path(bids_dir).resolve()
        output_dir = Path(output_dir).resolve()

        # Build Docker command
        cmd = [
            'docker', 'run', '--rm', '-it',
            '-v', f'{bids_dir}:/input:ro',
            '-v', f'{output_dir}:/output',
            '-v', f'{self.fs_license}:/license:ro',
        ]

        # Add GPU support if needed
        if device == 'gpu':
            cmd.extend(['--gpus', 'all'])

        cmd.append(self.docker_image)

        # Add DeepPrep arguments
        cmd.extend(['/input', '/output', 'participant'])
        cmd.extend(['--fs-license-file', '/license'])
        cmd.extend(['--nthreads', str(nthreads)])
        cmd.extend(['--device', device])

        if participant_labels:
            cmd.extend(['--participant-label'] + participant_labels)

        if anat_only:
            cmd.append('--anat-only')

        if skip_bids_validation:
            cmd.append('--skip-bids-validation')

        # Run command
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        else:
            print(f"Success: {result.stdout}")

        return result

# Usage example
runner = DeepPrepRunner(
    docker_image='pbfslab/deepprep:25.1.0',
    fs_license_path='/license/license.txt'
)

# Process single subject
result = runner.run(
    bids_dir='/data/bids',
    output_dir='/data/output',
    participant_labels=['001'],
    nthreads=8
)

# Check if successful
if result.returncode == 0:
    print("Processing completed successfully!")
```

### Batch Processing with Progress Tracking

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

class DeepPrepBatchProcessor:
    def __init__(self, runner: DeepPrepRunner):
        self.runner = runner
        self.logger = logging.getLogger(__name__)

    def process_subject(self, subject_id, bids_dir, output_dir, **kwargs):
        """Process a single subject"""
        try:
            result = self.runner.run(
                bids_dir=bids_dir,
                output_dir=output_dir,
                participant_labels=[subject_id],
                **kwargs
            )
            return {
                'subject_id': subject_id,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except Exception as e:
            self.logger.error(f"Failed to process {subject_id}: {e}")
            return {
                'subject_id': subject_id,
                'success': False,
                'error': str(e)
            }

    def process_batch(self,
                      subject_ids: List[str],
                      bids_dir: str,
                      output_dir: str,
                      max_workers: int = 4,
                      **kwargs):
        """
        Process multiple subjects in parallel

        Args:
            subject_ids: List of subject IDs
            bids_dir: BIDS directory
            output_dir: Output directory
            max_workers: Number of parallel processes
            **kwargs: Additional arguments to pass to runner.run()
        """
        results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_subject,
                    subj,
                    bids_dir,
                    output_dir,
                    **kwargs
                ): subj for subj in subject_ids
            }

            # Progress bar
            with tqdm(total=len(subject_ids), desc="Processing subjects") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

                    status = "✓" if result['success'] else "✗"
                    pbar.set_postfix({'last': f"{status} {result['subject_id']}"})
                    pbar.update(1)

        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\nCompleted: {successful}/{len(subject_ids)} subjects")

        # Save log
        import json
        log_file = Path(output_dir) / 'processing_log.json'
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Log saved to: {log_file}")

        return results

# Usage
runner = DeepPrepRunner(fs_license_path='/license/license.txt')
batch_processor = DeepPrepBatchProcessor(runner)

subjects = ['001', '002', '003', '004', '005']
results = batch_processor.process_batch(
    subject_ids=subjects,
    bids_dir='/data/bids',
    output_dir='/data/output',
    max_workers=2,  # Run 2 subjects in parallel
    nthreads=4,     # Each uses 4 threads
    anat_only=False
)
```

### Singularity Version (for HPC)

```python
class DeepPrepSingularityRunner:
    def __init__(self, sif_path: str, fs_license_path: str):
        """
        Args:
            sif_path: Path to DeepPrep Singularity image (.sif)
            fs_license_path: Path to FreeSurfer license
        """
        self.sif_path = Path(sif_path)
        self.fs_license = Path(fs_license_path)

        if not self.sif_path.exists():
            raise FileNotFoundError(f"Singularity image not found: {sif_path}")

    def run(self, bids_dir, output_dir, participant_labels=None, **kwargs):
        """Run DeepPrep with Singularity"""
        bids_dir = Path(bids_dir).resolve()
        output_dir = Path(output_dir).resolve()

        cmd = [
            'singularity', 'run',
            '--bind', f'{bids_dir}:/input',
            '--bind', f'{output_dir}:/output',
            '--bind', f'{self.fs_license}:/license',
            str(self.sif_path),
            '/input', '/output', 'participant',
            '--fs-license-file', '/license'
        ]

        # Add participant labels
        if participant_labels:
            cmd.extend(['--participant-label'] + participant_labels)

        # Add other options
        for key, value in kwargs.items():
            if isinstance(value, bool) and value:
                cmd.append(f'--{key.replace("_", "-")}')
            else:
                cmd.extend([f'--{key.replace("_", "-")}', str(value)])

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        return result

# Usage on HPC
runner = DeepPrepSingularityRunner(
    sif_path='/apps/deepprep-25.1.0.sif',
    fs_license_path='/home/user/license.txt'
)

result = runner.run(
    bids_dir='/scratch/data/bids',
    output_dir='/scratch/data/output',
    participant_labels=['001'],
    nthreads=16,
    device='cpu'
)
```

### Smart Pipeline with Auto-retry

```python
import time
from datetime import datetime

class SmartDeepPrepRunner(DeepPrepRunner):
    def run_with_retry(self, max_retries=3, retry_delay=60, **kwargs):
        """
        Run DeepPrep with automatic retry on failure

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries (seconds)
            **kwargs: Arguments passed to run()
        """
        for attempt in range(max_retries):
            print(f"\n[{datetime.now()}] Attempt {attempt + 1}/{max_retries}")

            result = self.run(**kwargs)

            if result.returncode == 0:
                print("✓ Success!")
                return result

            print(f"✗ Failed with code {result.returncode}")

            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        raise RuntimeError(f"Failed after {max_retries} attempts")

    def run_with_validation(self, bids_dir, output_dir, participant_labels, **kwargs):
        """
        Run with automatic output validation
        """
        # Run preprocessing
        result = self.run(
            bids_dir=bids_dir,
            output_dir=output_dir,
            participant_labels=participant_labels,
            **kwargs
        )

        if result.returncode != 0:
            return result

        # Validate outputs
        output_path = Path(output_dir) / 'deepprep'

        for subj_id in participant_labels:
            subj_dir = output_path / f'sub-{subj_id}'

            # Check key files
            required_files = [
                subj_dir / 'anat' / f'sub-{subj_id}_space-MNI152_T1w.nii.gz',
                output_path / f'sub-{subj_id}.html'
            ]

            for file in required_files:
                if not file.exists():
                    print(f"✗ Missing output: {file}")
                    result.returncode = 1
                    return result

        print("✓ All outputs validated")
        return result

# Usage
smart_runner = SmartDeepPrepRunner(fs_license_path='/license/license.txt')

# Automatic retry on failure
result = smart_runner.run_with_retry(
    bids_dir='/data/bids',
    output_dir='/data/output',
    participant_labels=['001'],
    max_retries=3,
    nthreads=8
)

# With validation
result = smart_runner.run_with_validation(
    bids_dir='/data/bids',
    output_dir='/data/output',
    participant_labels=['001', '002'],
    nthreads=8
)
```

### Integration Example: Full Workflow

```python
def preprocess_and_load(subject_ids, bids_dir, output_dir, fs_license):
    """
    Complete workflow: preprocess → QC → load data

    Returns:
        Dictionary with preprocessed data and QC metrics
    """
    # 1. Run DeepPrep
    runner = DeepPrepRunner(fs_license_path=fs_license)
    batch = DeepPrepBatchProcessor(runner)

    print("Step 1: Running DeepPrep...")
    results = batch.process_batch(
        subject_ids=subject_ids,
        bids_dir=bids_dir,
        output_dir=output_dir,
        max_workers=2,
        nthreads=4
    )

    # 2. Quality control
    print("\nStep 2: Quality control...")
    qc_results = []
    for subj_id in subject_ids:
        try:
            metrics = extract_qc_metrics(output_dir, subj_id)
            qc_results.append(metrics)
        except Exception as e:
            print(f"QC failed for {subj_id}: {e}")

    qc_df = pd.DataFrame(qc_results)
    passing_subjects = qc_df[qc_df['fd_mean'] < 0.5]['subject_id'].tolist()

    print(f"QC passed: {len(passing_subjects)}/{len(subject_ids)}")

    # 3. Load preprocessed data
    print("\nStep 3: Loading preprocessed data...")
    dataset = {}

    for subj_id in passing_subjects:
        try:
            loader = DeepPrepLoader(output_dir, subj_id)
            t1_data, affine = loader.load_t1()

            dataset[subj_id] = {
                'data': t1_data,
                'affine': affine,
                'qc_metrics': qc_df[qc_df['subject_id'] == subj_id].to_dict('records')[0]
            }
        except Exception as e:
            print(f"Failed to load {subj_id}: {e}")

    print(f"\nLoaded {len(dataset)} subjects")
    return dataset

# Usage: One-line preprocessing and loading
subjects = ['001', '002', '003']
dataset = preprocess_and_load(
    subject_ids=subjects,
    bids_dir='/data/bids',
    output_dir='/data/output',
    fs_license='/license/license.txt'
)

# Access data
for subj_id, data in dataset.items():
    print(f"{subj_id}: shape={data['data'].shape}, FD={data['qc_metrics']['fd_mean']:.3f}")
```


