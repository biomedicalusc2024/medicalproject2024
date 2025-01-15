# Medical Project

dataloader idea:
    load every dataset in following format: self.trainset = [[source_1, target_1], ..., [source_n, target_n]] where source_i = [feature_1, ..., feature_m] containing all source related feature and target_i = [feature_1, ..., feature_k] containing all target related feature.(or just one list containing all features)

    explain in print_stats what each position in source and target represent, what subdataset(train, test, val or only all) are supported and what format are supported.

    maybe maintaining a list in README on which datasets are done and if challenged, what is the problem.

    maybe we need some rule on naming and reference of dataset

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
    VisualGrouping/ChestX_ray8: dataEntity and split file has no direct urls, data resource can be downloaded.
    MolecularGeneration/CrossDocked2020: Dataset is too big and too much files, download is done but need advice on how to provide to user.
    summerization/TREC: need explanation on loading which part of xml file.