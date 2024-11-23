import copy
from dataset import ImgDataset, TxtDataset, collate_fn
from func_to_script import script

from typing import Callable
from pathlib import Path
import numpy as np
import pandas as pd

from timm import create_model
import torch
import clip
import json

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# def train_model(
#     meta_df: pd.DataFrame,
#     data_fldr: str,
#     img_fldr: str,
#     num_classes: int,
#     num_epochs: int = 20,
#     min_size: int = 60,
#     valid_frac: float = 0.1,
#     collate_fn: Callable = None,
# ):

#     model = create_model("resnetrs50", pretrained=True, num_classes=num_classes)

#     # train/validation split
#     valid_row_indexes = meta_df.sample(frac=valid_frac).index
#     meta_df["is_valid"] = False
#     meta_df.loc[valid_row_indexes, "is_valid"] = True

#     train_ds = MetadataDataset(
#         meta_df.query("is_valid == False"), img_fldr, get_train_transforms(min_size)
#     )

#     eval_ds = MetadataDataset(
#         meta_df.query("is_valid == True"), img_fldr, get_eval_transforms(min_size)
#     )

#     trainer = train_classification_model(
#         model,
#         train_ds=train_ds,
#         eval_ds=eval_ds,
#         num_epochs=num_epochs,
#         outputs_path=data_fldr,
#         collate_fn=collate_fn,
#     )

#     return trainer.model


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
    """
    Creates a numpy array containing a 2048 element vector for each image in the folder specified by img_fldr.
    It trains a classification model on the images using the labels in the metadata.csv file.
    It then outputs the vectors from the classification model backbone.
    The output file contains a numpy array consisting of: [num_images, 2048]

    :param data_fldr: The location of the metadata.csv file and the location to save the output file
    :param img_fldr: The location of the images
    :param use_imagenet: Don't train the model, just use the imagenet weights - usually leads to worse results
    :param num_epochs: The number of times to train the dataset
    :param min_size: The minimum size of image
    :param device: The torch device to use ['cpu'|'cuda']
    """

    data_fldr = Path(data_fldr)
    img_fldr = Path(img_fldr)

    device = torch.device(device)

    # meta_df = pd.read_csv(data_fldr / "metadata.csv")

    # num_classes = meta_df.class_id.unique().shape[0]
    

    # check if need to update class_ids to ensure class_id >=0 && < num_classes as model training requires it
    # if meta_df["class_id"].max() > num_classes:
    #     existing_classes = sorted(meta_df.class_id.unique())
    #     new_mapping = {k: idx for idx, k in enumerate(existing_classes)}
    #     meta_df["class_id"] = meta_df.apply(lambda r: new_mapping[r.class_id], axis=1)

    # if use_imagenet:
    #     model = create_model("resnetrs50", pretrained=True)
    # else:
    #     model = train_model(
    #         meta_df,
    #         data_fldr,
    #         img_fldr,
    #         num_classes,
    #         num_epochs,
    #         min_size,
    #         collate_fn=collate_fn,
    #     )

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    # strip off the classification head (for a resnet50)
    # as we just want the backbone outputs
    # model.to(device=device)
    # emb_model = copy.deepcopy(model)
    # emb_model.fc = torch.nn.Identity(2048)
    # emb_model = emb_model.to(device)
    
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
    # whole dataset with eval
    # preds_ds = MetadataDataset(meta_data['annotations'], img_fldr, preprocess)
    img_ds = ImgDataset(img_names, img_fldr, preprocess)
    dl = DataLoader(
        img_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=6,
    )
    preds_img_features = None
    for idx, images in enumerate(tqdm(dl)):
        images = images.to(device)
        with torch.no_grad():
            batch_output = model.encode_image(images)
        
        batch_features = batch_output.cpu().numpy()
        if preds_img_features is None:
            preds_img_features = batch_features
        else:
            preds_img_features = np.concatenate((preds_img_features, batch_features), axis=0)
    np.save(data_fldr / "image_vectors_coco_val2017.npy", preds_img_features)
    with open(data_fldr / "image_vectors_filename.json", 'w') as fp:
        json.dump(img_names, fp)
    
    txt_ds = TxtDataset(captions)
    dl = DataLoader(
        txt_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=6,
    )
    preds_txt_features = None
    for idx, texts in enumerate(tqdm(dl)):
        texts = texts.to(device)
        with torch.no_grad():
            text_features = model.encode_text(texts)
        
        text_features = text_features.cpu().numpy()
        if preds_txt_features is None:
            preds_txt_features = text_features
        else:
            preds_txt_features = np.concatenate((preds_txt_features, text_features), axis=0)
    np.save(data_fldr / "text_vectors_coco_val2017.npy", preds_txt_features)
    with open(data_fldr / "text_vectors_captions.json", 'w') as fp:
        json.dump(captions, fp)

if __name__ == "__main__":
    main()
