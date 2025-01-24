# Medical Project

dataloader idea:
load every dataset in following format: self.trainset = [{"img_path": str, "mask_path": str, ...}, ...]

    maybe maintaining a list in README on which datasets are done and if challenged, what is the problem.

    maybe add a file or something to mark everything is downloaded and no need to download again next time used

problems:
Detection/Breast-Cancer-Screening-DBT: training set over 1TB, test and val over 100GB, hard to implement and test locally.
NamedEntityRecognition/DDIExtraction2013: not clear how to use data downloaded, need further explanation
QuestionAnswering/LiveQA_PREC_2017: training set are deleted on github, only test set left.
Summerization/PubMed: Error when loading dataset using datasets from huggingface, not clear how to fix
QuestionAnswering/RadQA: some dataset need registeration to download.
QuestionAnswering/CliCR: need to send request email to acquire link for data.
QuestionAnswering/CT_RATE: need to access token on huggingface.
QuestionAnswering/LLaVA_Med: not clear how to use data downloaded, need further explanation
QuestionAnswering/QUILT-1M: limited access.
QuestionAnswering/WSI_VQA: img resource not provided, need to download manually.
Caption/PadChest: need requests.
VisualGrouping/ChestX_ray8: dataEntity and split file has no direct urls, data resource can be downloaded. Seem same with detaction/ChestXRays.
MolecularGeneration/CrossDocked2020: Dataset is too big and too much files, download is done but need advice on how to provide to user.
summerization/TREC: need explanation on loading which part of xml file.
Segmentation/LIDC-IDRI: requiring downloading NBIA Data Retriever
segmentation/AbdomenCT-1K: require permission
Classification/MIMIC-CXR: require permission and training
Caption/FFA-IR: require permission
Caption/MedICaT: need to fill out form to get access
Detection/Breast-Cancer-Screening-DBT: data missing
Mask Language/Radiation Oncology Literature: data missing
others/MIMIC-CXR Radiology Reports: data missing
Classification/Pitt: data missing
