# Medical Project

## dataloader:
    caption: ROCO, IUXray, PMC_OA

    classification: HoC, ROND, PTB_XL, Cirrhosis, ChestXRays, CheXpert_small, StrokePrediction, HepatitisCPrediction, HeartFailurePrediction, NHANES, IS_A, MedMnist, NHIS, MEPS

    database: CORD19

    detection: ChestXRays

    inference: ROND, BioNLI, NHIS, MEPS

    molecular generation: MOSES, CrossDocked2020

    named entity recognition: ROND, SourceData, DDIEtraction2013

    prediction: NHIS, MEPS

    question answering: ROND, VQA_RAD, PMC_VQA, MedMCQA, MedQA_USMLE, LiveQA_TREC_2017, MedicationQA, LLaVA_Med, Path_VQA, WSI_VQA, PubMedQA

    relation extraction: BC5CDR

    segmentation: ACDC, BraTS, BUID, CIR, Kvasir, Pancreas, ISIC_2018, ISIC_2019, LA, LiTS, Hippo, ChestXray, MSD, Covid_QU_EX, CheXmask, SIIM_ACR_Pneumothorax, CBIS_DDSM

    summerization: TREC, MeQSum

    time series: ExtMarker

    virtual screening: ZINC

    visual grouping: SLAKE, ChestX_ray8

## preprocess:
    deduplication: QA, BioMed

    JERS: Brain

    clinical_trail data

    patient data

    3D Image & quality control

## evaluation:
    dice_coef

    dice_accuracy

    dice_loss

    iou

    hausdorff_distance

    classification_metrics_sklearn

    calculate_map_50

    accuracy

    sensitivity

    specificity

    mean_edge_error

    mean_absolute_error

    get_boundary_region

    closed_ended_accuracy

    open_ended_accuracy

    overall_accuracy

    region_specific_auc

    TRECEvaluator:
        precision_at_k
        average_precision
        ndcg

    ndcg_at_k

    ndcd_at_k

    bleu_score

    exact_match

    meteor_score

    rouge_acc

    factent_score

    miou

    logp_score

    qed_score

    sa_score

    array_to_mol