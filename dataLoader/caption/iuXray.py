import os
import json
import numpy
import random
import warnings
import pandas as pd
from shutil import rmtree
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

from ..utils import print_sys


def getIUXray(path):
    try:
        dataPath = os.path.join(path, "IUXray")
        train_path = os.path.join(dataPath, "train_images.tsv")
        test_path = os.path.join(dataPath, "test_images.tsv")
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            downloadData(dataPath)
            splitFile(dataPath)
        return loadLocalFiles(dataPath)
    except Exception as e:
        print_sys(f"error: {e}")


def split_cases(reports_images, reports_text, keys, filename):
    new_images = {}

    for key in keys:
        for image in reports_images[key]:
            new_images[image] = reports_text[key]

    with open(filename, "w") as output_file:
        for new_image in new_images:
            output_file.write(new_image + "\t" + new_images[new_image])
            output_file.write("\n")


def downloadData(dataPath):
    try:
        rmtree(dataPath)
    except BaseException:
        pass

    os.makedirs(dataPath, exist_ok=True)

    # download PNG images
    os.system(f"wget -P {dataPath} https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz")

    # download reports
    os.system(f"wget -P {dataPath} https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz")

    # create folder for images
    os.makedirs(os.path.join(dataPath, "iu_xray_images"))

    # unzip
    os.system(f"tar -xzf {os.path.join(dataPath, 'NLMCXR_png.tgz')} -C {os.path.join(dataPath, 'iu_xray_images')}")
    os.system(f"tar -xzf {os.path.join(dataPath, 'NLMCXR_reports.tgz')} -C {dataPath}")

    os.remove(os.path.join(dataPath, 'NLMCXR_png.tgz'))
    os.remove(os.path.join(dataPath, 'NLMCXR_reports.tgz'))


def splitFile(dataPath):
    # read the reports xml files and create the dataset tsv
    reports_path = os.path.join(dataPath, "ecgen-radiology")

    reports = os.listdir(reports_path)

    reports.sort()

    reports_with_no_image = []
    reports_with_empty_sections = []
    reports_with_no_impression = []
    reports_with_no_findings = []

    images_captions = {}
    images_major_tags = {}
    images_auto_tags = {}
    reports_with_images = {}
    text_of_reports = {}

    for report in reports:
        tree = ET.parse(os.path.join(reports_path, report))
        root = tree.getroot()
        img_ids = []
        # find the images of the report
        images = root.findall("parentImage")
        # if there aren't any ignore the report
        if len(images) == 0:
            reports_with_no_image.append(report)
        else:
            sections = root.find("MedlineCitation").find("Article").find("Abstract").findall("AbstractText")
            # find impression and findings sections
            for section in sections:
                if section.get("Label") == "FINDINGS":
                    findings = section.text
                if section.get("Label") == "IMPRESSION":
                    impression = section.text

            if impression is None and findings is None:
                reports_with_empty_sections.append(report)
            else:
                if impression is None:
                    reports_with_no_impression.append(report)
                    caption = findings
                elif findings is None:
                    reports_with_no_findings.append(report)
                    caption = impression
                else:
                    caption = impression + " " + findings

                # get the MESH tags
                tags = root.find("MeSH")
                major_tags = []
                auto_tags = []
                if tags is not None:
                    major_tags = [t.text for t in tags.findall("major")]
                    auto_tags = [t.text for t in tags.findall("automatic")]

                for image in images:
                    iid = image.get("id") + ".png"
                    images_captions[iid] = caption
                    img_ids.append(iid)
                    images_major_tags[iid] = major_tags
                    images_auto_tags[iid] = auto_tags

                reports_with_images[report] = img_ids
                text_of_reports[report] = caption

    print("Found", len(reports_with_no_image), "reports with no associated image")
    print("Found", len(reports_with_empty_sections), "reports with empty Impression and Findings sections")
    print("Found", len(reports_with_no_impression), "reports with no Impression section")
    print("Found", len(reports_with_no_findings), "reports with no Findings section")

    print("Collected", len(images_captions), "image-caption pairs")

    with open(os.path.join(dataPath, "iu_xray.tsv"), "w") as output_file:
        for image_caption in images_captions:
            output_file.write(image_caption + "\t" + images_captions[image_caption])
            output_file.write("\n")

    # Safer JSON storing
    with open(os.path.join(dataPath, "iu_xray_captions.json"), "w") as output_file:
        output_file.write(json.dumps(images_captions))
    with open(os.path.join(dataPath, "iu_xray_major_tags.json"), "w") as output_file:
        output_file.write(json.dumps(images_major_tags))
    with open(os.path.join(dataPath, "iu_xray_auto_tags.json"), "w") as output_file:
        output_file.write(json.dumps(images_auto_tags))

    # perform a case based split
    random.seed(42)
    keys = list(reports_with_images.keys())
    random.shuffle(keys)

    train_split = int(numpy.floor(len(reports_with_images) * 0.9))

    train_keys = keys[:train_split]
    test_keys = keys[train_split:]

    train_path = os.path.join(dataPath, "train_images.tsv")
    test_path = os.path.join(dataPath, "test_images.tsv")

    split_cases(reports_with_images, text_of_reports, train_keys, train_path)
    split_cases(reports_with_images, text_of_reports, test_keys, test_path)


def loadLocalFiles(dataPath):
    train_path = os.path.join(dataPath, "train_images.tsv")
    test_path = os.path.join(dataPath, "test_images.tsv")
    df_train = pd.read_csv(train_path, sep='\t', names=["source", "target"])
    df_test = pd.read_csv(test_path, sep='\t', names=["source", "target"])
    dataset_train = {
        "source": [[s] for s in df_train["source"].to_numpy().tolist()],
        "target": [[s] for s in df_train["target"].to_numpy().tolist()],
    }
    dataset_test = {
        "source": [[s] for s in df_test["source"].to_numpy().tolist()],
        "target": [[s] for s in df_test["target"].to_numpy().tolist()],
    }
    return dataset_train, dataset_test