import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import transformers

class ImgDataset(Dataset):
    def __init__(
        self, img_names, img_dir
    ):
        self.img_names = img_names
        self.img_dir = img_dir
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path)

        return image

class TxtDataset(Dataset):
    def __init__(
        self, captions
    ):
        self.captions = captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]

        return caption


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""
    img_prompt: str
    txt_prompt: str
    processor: transformers.LlavaNextProcessor
    # device:torch.device
    # transform:DataAugmentation

    def __call__(self, instances):
        if isinstance(instances[0], str):
            batch = self.processor([self.txt_prompt.replace('<sent>', text) for text in instances], return_tensors="pt", padding=True)
        else:
            batch = self.processor([self.img_prompt]*len(instances), instances, return_tensors="pt", padding=True)

        return batch
