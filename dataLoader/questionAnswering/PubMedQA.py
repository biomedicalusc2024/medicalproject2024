import os
import json
import math
import random
random.seed(0)
import warnings
import pandas as pd
from functools import reduce

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

PubMedQA_SUBTITLE = ["A", "U", "L"]


# tested by tjl 2025/1/19
def split(dataset, fold):
    '''
    dataset: dataset dict
    fold: number of splits

    output list of splited datasets

    Split the dataset for each label to ensure label proportion of different subsets are similar
    '''
    add = lambda x: reduce(lambda a, b: a+b, x)
    
    label2pmid = {'yes': [], 'no': [], 'maybe': []}
    for pmid, info in dataset.items():
        label2pmid[info['final_decision']].append(pmid)

    label2pmid = {k: split_label(v, fold) for k, v in label2pmid.items()} # splited

    output = []

    for i in range(fold):
        pmids = add([v[i] for _, v in label2pmid.items()])
        output.append({pmid: dataset[pmid] for pmid in pmids})

    if len(output[-1]) != len(output[0]): # imbalanced: [51, 51, 51, 51, 51, 51, 51, 51, 51, 41]
        # randomly pick one from each to the last
        for i in range(fold-1):
            pmids = list(output[i])
            picked = random.choice(pmids)
            output[-1][picked] = output[i][picked]
            output[i].pop(picked)

    return output

def split_label(pmids, fold):
    '''
    pmids: a list of pmids (of the same label)
    fold: number of splits

    output: list of split lists
    '''
    random.shuffle(pmids)

    num_all = len(pmids)
    num_split = math.ceil(num_all / fold)

    output = []
    for i in range(fold):
        if i == fold - 1:
            output.append(pmids[i*num_split: ])
        else:
            output.append(pmids[i*num_split: (i+1)*num_split])

    return output

def combine_other(cv_sets, fold):
    '''
    combine other cv sets
    '''
    output = {}

    for i in range(10):
        if i != fold:
            for pmid, info in cv_sets[i].items():
                output[pmid] = info

    return output

def split_dataset(datasetPath):
    dataset = json.load(open(datasetPath))
    
    pmids = list(dataset)
    random.shuffle(pmids)

    train_split = {pmid: dataset[pmid] for pmid in pmids[:200000]}
    dev_split = {pmid: dataset[pmid] for pmid in pmids[200000:]}

    with open('../data/pqaa_train_set.json', 'w') as f:
        json.dump(train_split, f, indent=4)
    with open('../data/pqaa_dev_set.json', 'w') as f:
        json.dump(dev_split, f, indent=4)

def getPubMedQA(path, subtitle):
    urls = [
        "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
        "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json",
        "https://drive.usercontent.google.com/download?id=1RsGLINVce-0GsDkCLDuLZmoLuzfmoCuQ&export=download&authuser=0&confirm=t&uuid=8923ed38-84cb-4317-addf-b635ce565ecf&at=APvzH3pEQfxsTGRJ4tK6igWNN5IC:1736300564806", # U
        "https://drive.usercontent.google.com/download?id=15v1x6aQDlZymaHGP7cZJZZYFfeJt2NdS&export=download&authuser=0&confirm=t&uuid=0dafa514-685b-4f1f-9a4d-694d78329986&at=APvzH3ovBfidpzSJZPgnE5Yo7GM5:1736301193668", # A
    ]
    
    return datasetLoad(urls=urls, subtitle=subtitle, path=path, datasetName="PubMedQA")


def datasetLoad(urls, subtitle, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath, subtitle)
        else:
            os.makedirs(datasetPath, exist_ok=True)

            download_file(urls[0], os.path.join(datasetPath, "ori_PQAL.json"))
            download_file(urls[1], os.path.join(datasetPath, "test_ground_truth.json"))
            download_file(urls[2], os.path.join(datasetPath, "ori_PQAU.json"))
            download_file(urls[3], os.path.join(datasetPath, "ori_PQAA.json"))
            
            return loadLocalFiles(datasetPath, subtitle)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path, subtitle):
    if subtitle == "A":
        df = pd.read_json(os.path.join(path, "ori_PQAA.json")).T
    elif subtitle == "L":
        df = pd.read_json(os.path.join(path, "ori_PQAL.json")).T
    elif subtitle == "U":
        df = pd.read_json(os.path.join(path, "ori_PQAU.json")).T
    else:
        raise AttributeError(f'Please enter dataset name in PubMedQA-subset format and select the subsection of PubMedQA in {PubMedQA_SUBTITLE}')
    
    dataset = df.to_dict(orient='records')
    return dataset