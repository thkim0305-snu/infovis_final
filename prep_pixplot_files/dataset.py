import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import clip

class ImgDataset(Dataset):
    def __init__(
        self, img_names, img_dir, transform=None,
    ):
        self.img_names = img_names
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

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
        caption = clip.tokenize([caption])[0]

        return caption

def collate_fn(batch):

    # images, labels = zip(*batch)

    # images = torch.stack(images)
    # labels = torch.stack(labels)
    
    batch = torch.stack(batch)

    return batch
