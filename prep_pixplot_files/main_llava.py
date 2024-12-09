import copy
from dataset_llava import ImgDataset, TxtDataset, DataCollator
from func_to_script import script

from typing import Callable
from pathlib import Path
import numpy as np
import pandas as pd

from timm import create_model
import torch
import json

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

@script
def main(
    data_fldr: str,
    # annotations: str,
    img_fldr: str,
    use_imagenet: bool = False,
    num_epochs: int = 20,
    min_size: int = 60,
    device: str = "cuda",
):
    data_fldr = Path(data_fldr)
    img_fldr = Path(img_fldr)

    device = torch.device(device)
    
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'

    processor = LlavaNextProcessor.from_pretrained('royokong/e5-v')
    model = LlavaNextForConditionalGeneration.from_pretrained('royokong/e5-v', torch_dtype=torch.float16).cuda()

    img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
    text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')

    model.eval()
    with open(f'{data_fldr}/annotations/captions_val2017.json','r') as fp:
        meta_data = json.load(fp)
    
    img_names = []
    captions = []
    for item in meta_data['annotations']:
        captions.append(item['caption'])
        img_name = str(item['image_id']).zfill(12) + '.jpg'
        if img_name not in img_names:
            img_names.append(img_name)
    print(f'len images: {len(img_names)}')
    print(f'len texts: {len(captions)}')
    
    # img_ds = ImgDataset(img_names, img_fldr)
    # dl = DataLoader(
    #     img_ds,
    #     batch_size=8,
    #     shuffle=False,
    #     collate_fn=DataCollator(img_prompt, text_prompt, processor),
    #     num_workers=2,
    # )

    # preds_img_features = None
    # for idx, images in enumerate(tqdm(dl)):
    #     images = images.to(device)
    #     with torch.no_grad():
    #         batch_output = model(**images, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        
    #     batch_features = batch_output.cpu().numpy()
    #     if preds_img_features is None:
    #         preds_img_features = batch_features
    #     else:
    #         preds_img_features = np.concatenate((preds_img_features, batch_features), axis=0)
    # np.save(data_fldr / "image_vectors_coco_val2017_llava.npy", preds_img_features)
    # with open(data_fldr / "image_vectors_filename.json", 'w') as fp:
    #     json.dump(img_names, fp)
    
    txt_ds = TxtDataset(captions)
    dl = DataLoader(
        txt_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=DataCollator(img_prompt, text_prompt, processor),
        num_workers=2,
    )
    preds_txt_features = None
    for idx, texts in enumerate(tqdm(dl)):
        texts = texts.to(device)
        with torch.no_grad():
            text_features = model(**texts, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        
        text_features = text_features.cpu().numpy()
        if preds_txt_features is None:
            preds_txt_features = text_features
        else:
            preds_txt_features = np.concatenate((preds_txt_features, text_features), axis=0)
    np.save(data_fldr / "text_vectors_coco_val2017_llava.npy", preds_txt_features)
    with open(data_fldr / "text_vectors_captions.json", 'w') as fp:
        json.dump(captions, fp)

if __name__ == "__main__":
    main()
