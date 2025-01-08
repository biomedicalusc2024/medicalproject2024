# Medical Project

dataloader idea:
    load every dataset in following format: self.trainset = [[source_1, target_1], ..., [source_n, target_n]] where source_i = [feature_1, ..., feature_m] containing all source related feature and target_i = [feature_1, ..., feature_k] containing all target related feature.

    explain in print_stats what each position in source and target represent, what subdataset(train, test, val or only all) are supported and what format are supported.

    maybe maintaining a list in README on which datasets are done and if challenged, what is the problem.

    maybe we need some rule on naming and reference of dataset

problems:
    Detection/Breast-Cancer-Screening-DBT: training set over 1TB, test and val over 100GB, hard to implement and test locally.
    NamedEntityRecognition/DDIExtraction2013: not clear how to use data downloaded, need further explanation
    QuestionAnswering/LiveQA_PREC_2017: training set are deleted on github, only test set left.
    Summerization/PubMed: Error when loading dataset using datasets from huggingface, not clear how to fix