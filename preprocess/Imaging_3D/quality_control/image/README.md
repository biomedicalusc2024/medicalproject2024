<div align="center">

# 🧠 MRIQC: MRI Quality Control<br>Automated Quality Assessment for MRI Data

### *No-Reference Image Quality Metrics for Automatic Prediction of Quality and Visual Reporting of MRI Scans*

</div>

---

## 📋 Abstract

MRIQC is an open-source software tool that automatically extracts image quality metrics (IQMs) from structural (T1w, T2w) and functional MRI data. The software is designed to provide researchers and clinicians with an automated, robust, and reproducible way to assess the quality of their MRI data. MRIQC generates reports for MR datasets, facilitating both automated quality assessment and manual quality control procedures.


## 🚀 Getting Started

### 1. System Requirements

- **Operating System**: Linux, macOS, Windows
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: ~5GB for software + space for your data
- **Python**: 3.8+ (if installing via pip)

### 2. Installation

```bash
# Create a new conda environment
conda create -n mriqc python=3.9 -y
conda activate mriqc

# Install MRIQC
python -m pip install -U mriqc
```

### 3. Download Example Dataset

Download the example BIDS dataset from [Google Drive](https://drive.google.com/drive/folders/0B2JWN60ZLkgkMGlUY3B4MXZIZW8?resourcekey=0-EYVSOlRbxeFKO8NpjWWM3w):

```bash
# Create data directory
mkdir -p data
cd data

# Download and extract dataset from Google Drive
# e.g. After downloading ds003.tar from the link above:
tar -xvf ds003.tar
```

### 4. Quick Start

**🚀 Basic Usage Example**

```bash
mriqc /path/to/data/ds003 /path/to/output participant

# Process specific participant
mriqc /path/to/data/ds003 /path/to/output participant \
    --participant-label S01 S02 S03

# Generate group-level report
mriqc /path/to/data/ds003 /path/to/output group
```

## 5. Data Format

### BIDS Dataset Structure

MRIQC requires data organized according to the [Brain Imaging Data Structure (BIDS)](https://www.nipreps.org/apps/framework/) specification:

```
ds000003/
 ├─ CHANGES
 ├─ dataset_description.json
 ├─ participants.tsv
 ├─ README
 ├─ sub-01/
 │ ├─ anat/
 │ │ ├─ sub-01_inplaneT2.nii.gz
 │ │ └─ sub-01_T1w.nii.gz
 │ └─ func/
 │ ├─ sub-01_task-rhymejudgment_bold.nii.gz
 │ └─ sub-01_task-rhymejudgment_events.tsv
 ├─ sub-02/
 ├─ sub-03/
```

## 📚 Citation

If you use MRIQC in your research, please cite:

```bibtex
@article{esteban2017mriqc,
  title={MRIQC: Advancing the automatic prediction of image quality in MRI from unseen sites},
  author={Esteban, Oscar and Birman, Daniel and Schaer, Marie and Koyejo, Oluwasanmi O and Poldrack, Russell A and Gorgolewski, Krzysztof J},
  journal={PloS one},
  volume={12},
  number={9},
  pages={e0184661},
  year={2017},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

## 🙏 Acknowledgements

We gratefully acknowledge:
- **[MRIQC](https://github.com/nipreps/mriqc)**
