# Medical Project 2024

A comprehensive medical data processing and machine learning framework designed for various healthcare applications including medical imaging, clinical data analysis, and predictive modeling.

## Overview

This project provides a modular architecture for handling diverse medical data types and implementing various machine learning tasks in the healthcare domain. The framework supports multiple data modalities and offers specialized loaders and processors for different medical applications.

## Project Structure

```
medicalproject2024/
├── __init__.py                 # Package initialization
├── requirement.yml             # Conda environment dependencies
├── dataLoader/                 # Data loading modules
│   ├── __init__.py
│   ├── baseLoader.py          # Base data loader class
│   ├── utils.py               # Utility functions
│   ├── caption/               # Medical image captioning
│   ├── classification/        # Medical classification tasks
│   ├── database/              # Database connection utilities
│   ├── detection/             # Medical object detection
│   ├── inference/             # Model inference utilities
│   ├── molecularGeneration/   # Molecular structure generation
│   ├── namedEntityRecognition/ # Medical NER tasks
│   ├── prediction/            # Medical prediction models
│   ├── questionAnswering/     # Medical Q&A systems
│   ├── relationExtraction/    # Medical relation extraction
│   ├── segmentation/          # Medical image segmentation
│   ├── summerization/         # Medical text summarization
│   ├── timeSeries/            # Medical time series analysis
│   ├── virtualScreening/      # Drug virtual screening modules
│   └── visualGrouping/        # Modules for grouping medical visuals
├── defaultData/               # Default datasets for testing and examples
│   ├── 3D_imaging/           # Sample 3D medical imaging datasets
│   ├── meps/                 # Medical Expenditure Panel Survey data
│   └── nhis/                 # National Health Interview Survey data
├── preprocess/               # Data preprocessing modules
│   ├── clinical_trial/       # Preprocessing clinical trial datasets
│   ├── deduplication/        # Removing duplicate records from datasets
│   ├── Imaging_3D/           # Preprocessing 3D medical imaging data
│   ├── JERS/                 # Preprocessing brain 3D imaging data
│   └── patient/              # Preprocessing patient data
├── tutorial/                 # Documentation and tutorials for the framework
└── utils/                    # General utility functions for the project
```

## Features

### Data Loaders
- **Multi-modal Support**: Handles various medical data types including imaging, text, and structured data
- **Specialized Loaders**: Task-specific data loaders for different medical applications
- **Database Integration**: Built-in database connectivity for medical databases

### Supported Tasks

- **Caption Generation**
    1. ROCO - Radiology Objects in COntext dataset for medical image captioning
    2. IUXray - Indiana University Chest X-ray collection with reports
    3. PMC_OA - PubMed Central Open Access subset for biomedical image captioning

- **Classification**
    1. Clinical Data - HoC, ROND, Cirrhosis, StrokePrediction, HepatitisCPrediction, HeartFailurePrediction, NHANES, IS_A, NHIS, MEPS
    2. Medical Imaging - PTB_XL, ChestXRays, CheXpert_small, MedMnist
    3. Chest X-ray Analysis - Comprehensive chest radiograph classification

- **Detection & Segmentation**
    1. Medical Object Detection - ChestXRays detection capabilities
    2. Multi-organ Segmentation - ACDC (cardiac), BraTS (brain tumors), Pancreas, LiTS (liver), Hippo (hippocampus)
    3. Medical Imaging - BUID, CIR, Kvasir, ISIC_2018/2019 (skin lesions), LA, MSD, ChestXray
    4. Specialized Segmentation - Covid_QU_EX, CheXmask, SIIM_ACR_Pneumothorax, CBIS_DDSM (breast imaging)

- **Natural Language Processing**
    1. Named Entity Recognition - ROND, SourceData, DDIEtraction2013
    2. Relation Extraction - BC5CDR for biomedical concept relations
    3. Text Summarization - TREC, MeQSum for medical document summarization
    4. Inference Tasks - ROND, BioNLI for natural language inference

- **Question Answering**
    1. Visual QA - VQA_RAD, PMC_VQA, LLaVA_Med, Path_VQA, WSI_VQA
    2. Medical QA - MedMCQA, MedQA_USMLE, LiveQA_TREC_2017, MedicationQA, PubMedQA
    3. General Medical - ROND question answering capabilities

- **Molecular & Drug Discovery**
    1. Molecular Generation - MOSES, CrossDocked2020 for drug discovery
    2. Virtual Screening - ZINC database integration
    3. Chemical Analysis - Advanced molecular structure analysis

- **Time Series & Prediction**
    1. Clinical Prediction - NHIS, MEPS for healthcare outcomes
    2. Biomarker Analysis - ExtMarker for temporal medical data
    3. Healthcare Forecasting - Longitudinal patient data analysis

- **Database & Visual Grouping**
    1. Research Database - CORD19 for COVID-19 research
    2. Visual Grouping - SLAKE, ChestX_ray8 for medical image organization

### Default Datasets
- **3D Imaging**: Sample 3D medical imaging datasets
- **MEPS**: Medical Expenditure Panel Survey data
- **NHIS**: National Health Interview Survey data

### Preprocessing Capabilities
- **Deduplication**: Advanced deduplication for QA datasets and BioMed data
- **JERS**: Specialized preprocessing for Brain imaging data
- **Clinical Trial Data**: Comprehensive clinical trial dataset preprocessing
- **Patient Data**: Structured patient information preprocessing
- **3D Imaging & Quality Control**: Advanced 3D medical image preprocessing with quality assurance

### Evaluation Metrics
- **Segmentation Metrics**: dice_coef, dice_accuracy, dice_loss, iou, hausdorff_distance, miou
- **Classification Metrics**: accuracy, sensitivity, specificity, classification_metrics_sklearn
- **Detection Metrics**: calculate_map_50, mean_edge_error, mean_absolute_error, get_boundary_region
- **QA Evaluation**: closed_ended_accuracy, open_ended_accuracy, overall_accuracy, exact_match
- **Information Retrieval**: TRECEvaluator with precision_at_k, average_precision, ndcg, ndcg_at_k, ndcd_at_k
- **Text Generation**: bleu_score, meteor_score, rouge_acc, factent_score
- **Molecular Evaluation**: logp_score, qed_score, sa_score, array_to_mol
- **Medical Imaging**: region_specific_auc for specialized medical image analysis

## Installation

1. Create a conda environment using the provided requirements:
```bash
conda env create -f requirement.yml
```

2. Activate the environment:
```bash
conda activate hugging-health
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage Examples
Refer to tutorial folder for usage

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Documentation

Detailed documentation and tutorials can be found in the [`tutorial/`](tutorial/) directory.

## License

This project is part of the USC Biobank initiative under the Hugging Health framework.

## Contact

For questions and support, please contact the development team through the USC Biobank project channels.