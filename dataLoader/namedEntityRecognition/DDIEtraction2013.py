import os
import warnings
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# tested by tjl 2025/1/31
def getDDIEtraction2013(path):
    urls = ["https://github.com/isegura/DDICorpus/raw/master/DDICorpus-2013.zip"]
    return datasetLoad(urls=urls, path=path, datasetName="DDIEtraction2013")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetZip = os.path.join(path, "DDICorpus-2013.zip")
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            download_file(urls[0], datasetZip, datasetPath)
            return loadLocalFiles(datasetPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    sources = ["DrugBank", "MedLine"]
    dataPath = os.path.join(path, "DDICorpus")
    trainDataPath = os.path.join(dataPath, "Train")
    testDataPath = os.path.join(dataPath, "Test", "Test for DrugNER task")

    trainset, testset = [], []
    for source in sources:
        trainPath = os.path.join(trainDataPath, source)
        xmls = [fn for fn in os.listdir(trainPath) if ".xml" in fn]
        for xml in xmls:
            xmlPath = os.path.join(trainPath, xml)
            xmlElement = {"fn": xml, "source": source}
            xmlElement["document"] = parse_xml(xmlPath)
            trainset.append(xmlElement)

    for source in sources:
        testPath = os.path.join(testDataPath, source)
        xmls = [fn for fn in os.listdir(testPath) if ".xml" in fn]
        for xml in xmls:
            xmlPath = os.path.join(testPath, xml)
            xmlElement = {"fn": xml, "source": source}
            xmlElement["document"] = parse_xml(xmlPath)
            testset.append(xmlElement)
    
    return trainset, testset
    

def parse_xml(xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    result = []

    for sentence in root:
        sentence_dict = {
            "text": sentence.attrib['text'].strip(),
        }
        text = sentence.attrib['text'].strip()
        entities = sentence.findall('entity')
        pairs = sentence.findall('pair')

        entity_list = []
        for entity in entities:
            entity_list.append({
                "charOffset": entity.attrib['charOffset'],
                "type": entity.attrib['type'],
                "text": entity.attrib['text'],
                "id": entity.attrib['id'],
            })
        sentence_dict["entities"] = entity_list

        pair_list = []
        for pair in pairs:
            pair_list.append({
                "ddi": pair.attrib['ddi'],
                "e2": pair.attrib['e2'],
                "e1": pair.attrib['e1'],
                "id": pair.attrib['id'],
            })
        sentence_dict["pairs"] = pair_list
        result.append(sentence_dict)
    return result