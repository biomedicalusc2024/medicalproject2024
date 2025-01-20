import os
import json
import shutil
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file


# need guidance on data needed
def getLLaVA_Med(path):
    urls = [
        'https://hanoverprod.z21.web.core.windows.net/med_llava/alignment/llava_med_alignment_500k.json',

        'https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_10k.json',
        'https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k.json',
        'https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k_inline_mention.json',
        'https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_fig_captions.json',

        'https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_eval_qa50_qa.jsonl',
        'https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_eval_qa50_fig_captions.json',
        'https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_qa50_instruct_caption_in_text_cleaned-60k-3epoch.json',

        'https://hanoverprod.z21.web.core.windows.net/med_llava/llava_med_image_urls.jsonl',
    ]
    return datasetLoad(urls, path=path, datasetName="LLaVA_Med")


def datasetLoad(urls, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        if os.path.exists(datasetPath):
            print_sys("Found local copy...")
            return loadLocalFiles(datasetPath)
        else:
            os.makedirs(os.path.join(datasetPath,'alignment'), exist_ok=True)
            os.makedirs(os.path.join(datasetPath,'instruct'), exist_ok=True)
            os.makedirs(os.path.join(datasetPath,'eval'), exist_ok=True)
            os.makedirs(os.path.join(datasetPath,'pmc_articles'), exist_ok=True)
            os.makedirs(os.path.join(datasetPath,'images'), exist_ok=True)

            for url in urls[:-1]:
                path, filename = url.split("/")[-2:]
                download_file(url, os.path.join(datasetPath,path,filename))

            img_urls = os.path.join(datasetPath, urls[-1].split("/")[-1])
            download_file(urls[-1], img_urls)
            input_data = []
            with open(img_urls) as f:
                for line in f:
                    input_data.append(json.loads(line))
            
            print('Downloading PMC articles')
            pmc_path = os.path.join(datasetPath,'pmc_articles')
            img_path = os.path.join(datasetPath,'images')
            for _, sample in enumerate(tqdm(input_data)):
                try:
                    download_file(sample['pmc_tar_url'], os.path.join(pmc_path, os.path.basename(sample['pmc_tar_url'])), pmc_path)
                    src = os.path.join(pmc_path, sample['image_file_path'])
                    dst = os.path.join(img_path, sample['pair_id']+'.jpg')
                    shutil.copyfile(src, dst)
                except Exception as e:
                    print('Error downloading PMC article: {}'.format(sample['pmc_tar_url']))
                    continue
        
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(path):
    breakpoint()