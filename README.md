# Medical Project

dataloader idea:
load every dataset in following format: self.trainset = [{"img_path": str, "mask_path": str, ...}, ...]

    maybe maintaining a list in README on which datasets are done and if challenged, what is the problem.

    maybe add a file or something to mark everything is downloaded and no need to download again next time used

problems:
Segmentation/CheXmask: data downloaded only contain mask data, source image need to be downloaded seperately.
Detection/Breast-Cancer-Screening-DBT: training set over 1TB, test and val over 100GB, hard to implement and test locally.
MolecularGeneration/CrossDocked2020: Currently use downsampled dataset, full dataset is over 60G which is too big
VisualGrouping/ChestX_ray8: dataEntity and split file has no direct urls, data resource can be downloaded. Seem same with detaction/ChestXRays.

Summerization/PubMed: Error when loading dataset using datasets from huggingface, not clear how to fix

Segmentation/LIDC-IDRI: requiring downloading NBIA Data Retriever
segmentation/AbdomenCT-1K: require permission
segmentation/BKAI_IGH_NeoPolyp: need to submit files to acquire permission
segmentation/NLST: dataset not public
segmentation/OCTA500: need account and password
Classification/MIMIC-CXR: require permission and training
Classification/Pitt: data missing
Caption/FFA-IR: require permission
Caption/MedICaT: need to fill out form to get access
Caption/PadChest: need requests.
Detection/Breast-Cancer-Screening-DBT: data missing
Mask Language/Radiation Oncology Literature: data missing
others/MIMIC-CXR Radiology Reports: data missing
QuestionAnswering/RadQA: some dataset need registeration to download.
QuestionAnswering/CliCR: need to send request email to acquire link for data.
QuestionAnswering/CT_RATE: need to access token on huggingface.
QuestionAnswering/QUILT-1M: limited access.
