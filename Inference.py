# setup pyvips

import os

os.system('ls /kaggle/input/pyvips-python-and-deb-package-gpu')
# intall the deb packages
os.system('yes | dpkg -i --force-depends /kaggle/input/pyvips-python-and-deb-package-gpu/linux_packages/archives/*.deb')
# install the python wrapper
os.system('pip install pyvips -f /kaggle/input/pyvips-python-and-deb-package-gpu/python_packages/ --no-index')


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
import gc
import time
from IPython import display
import glob
import random
from joblib import Parallel, delayed


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, datasets


test_df=pd.read_csv("/kaggle/input/UBC-OCEAN/test.csv")

print('test dataset:')
print(test_df.shape)
print(test_df.head())

TEST_IMG_DIR = "/kaggle/input/UBC-OCEAN/test_images/"
TEST_TBNLS_DIR = "/kaggle/input/UBC-OCEAN/test_thumbnails/"
TEST_TILES_DIR = "/kaggle/working/test_tiles/"


# tiling helper func

import cv2
import pyvips

os.environ['VIPS_CONCURRENCY'] = '4'
os.environ['VIPS_DISC_THRESHOLD'] = '15gb' #use disk caching instead of memory when the image exceeds 15GB

# extract tiles from an image
# critial to set up max_samples, because all samples will be used for predict

def extract_test_tiles(img_path, size=1792, scale=0.125, drop_thr=0.6, white_thr=240, max_samples=20):
    
    # print(f"processing: {img_path}")
    im = pyvips.Image.new_from_file(img_path) #load image
    
    w=h=size
    new_size = int(size * scale), int(size * scale)
    
    # https://stackoverflow.com/a/47581978/4521646
    idxs = [(y, y + h, x, x + w) for y in range(0, im.height, h) for x in range(0, im.width, w)]
    
    # random subsample
    max_samples = max_samples if isinstance(max_samples, int) else int(len(idxs) * max_samples)
    random.shuffle(idxs)
    
    images = []
    for y, y_, x, x_ in idxs:

        tile = im.crop(x, y, min(w, im.width - x), min(h, im.height - y)).numpy()[..., :3] 

        # increase tile size to (h,w) for edge tiles
        if tile.shape[:2] != new_size:
            tile_ = tile
            tile_size = (h, w) if tile.ndim == 2 else (h, w, tile.shape[2])
            tile = np.zeros(tile_size, dtype=tile.dtype)
            tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_
        
        # reduce size
        tile = np.array(Image.fromarray(tile).resize(new_size, Image.LANCZOS))
        
        # emptry ratio detection
        black_bg = np.sum(tile, axis=2) == 0
        tile[black_bg, :] = 255
        mask_bg = np.mean(tile, axis=2) > white_thr
        if np.sum(mask_bg) >= (np.prod(mask_bg.shape) * drop_thr):
            #print(f"skip almost empty tile: {k:06}_{int(x_ / w)}-{int(y_ / h)}")
            continue
        # print(tile.shape, tile.dtype, tile.min(), tile.max())
        
        images.append(tile)
        if len(images) >= max_samples:
            break
    return images


# standard image augmentaion procedures and color normalizations

#tile_color_mean=[0.8636166450980394, 0.7583822915468411, 0.8537942079084968]
#tile_color_std=[0.06693396412705377, 0.09844775155547589, 0.0531690918923438]

net_input=1024

import albumentations as A
from albumentations import Compose, CenterCrop, Normalize
from albumentations.pytorch import ToTensorV2

# use albumentations for augmentation
val_transform = A.Compose([
        CenterCrop(height=net_input, width=net_input),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()])


# given one sample's img_path, __getitem__ outputs one tile at a time

class ImageTilesDataset(Dataset):

    def __init__(self, img_path:str, size=2048, scale=0.5, drop_thr=0.6, max_samples=20, transform=None):
        assert os.path.isfile(img_path)
        self.transform = transform
        self.tiles=extract_test_tiles(img_path, size=size, scale=scale, drop_thr=drop_thr, max_samples=max_samples)
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int):
        tile = self.tiles[idx]
        if np.max(tile) < 1.5:
            tile = np.clip(tile * 255, 0, 255).astype(np.uint8)
        # augmentation
        if self.transform:
            augmented = self.transform(image=tile)  # 注意，这里输入的img是一个numpy数组
            tile = augmented['image']
        return tile


# explore devices

def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    devices=[torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_all_gpus())


# infer helper

import shutil
from torch.utils.data import DataLoader
import torch.nn.functional as F
from joblib.externals.loky.backend.context import get_context

label_dict_reverse = {0: 'CC', 1: 'EC', 2: 'HGSC', 3: 'LGSC', 4: 'MC'}

def infer_single_image(idx_row, nets, device=try_gpu(), max_samples=20, threshold=10) -> dict:
    row = dict(idx_row[1])
    
    # prepare data - cut and load tiles
    img_path = os.path.join(TEST_IMG_DIR, f"{str(row['image_id'])}.png")
    test_dataset = ImageTilesDataset(img_path, size=3072, scale=0.5, max_samples=max_samples, transform=val_transform)
    if not len(test_dataset):
        print (f"seem no tiles were cut for `{row['image_id']}`")
        return row
    test_iter = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, num_workers=2, shuffle=False,
        # https://github.com/pytorch/pytorch/issues/44687#issuecomment-790842173
        multiprocessing_context=get_context('loky'))

    # iterate over images and collect predictions
    preds1 = []
    preds2 = []
    preds3 = []
    for imgs in test_iter:
        #print(f"{imgs.shape}")
        with torch.no_grad():
            imgs = imgs.to(device).half()
            pred = nets[0](imgs)
            pred1 = F.softmax(pred, dim=1)
            
            pred = nets[1](imgs)
            pred2 = F.softmax(pred, dim=1)
            
            pred = nets[2](imgs)
            pred3 = F.softmax(pred, dim=1)
        preds1 += pred1.cpu().numpy().tolist()
        preds2 += pred2.cpu().numpy().tolist()
        preds3 += pred3.cpu().numpy().tolist()
    # print(f"Sum contrinution from all tiles: {np.sum(preds, axis=0)}")
    # print(f"Max contribution over all tiles: {np.max(preds, axis=0)}")
    
    # decide label
    preds1 = np.sum(preds1, axis=0)
    preds2 = np.sum(preds2, axis=0)
    preds3 = np.sum(preds3, axis=0)
    
    lb1 = np.argmax(preds1)
    lb1_prob = np.max(preds1)
    
    lb2 = np.argmax(preds2)
    lb2_prob = np.max(preds2)
    lb3 = np.argmax(preds3)
    lb3_prob = np.max(preds3)
    
    labels = [lb1, lb2, lb3]
    print(labels, lb1_prob)
    lb = labels[np.argmax([lb1_prob, lb2_prob, lb3_prob])]
        
    row['label'] = label_dict_reverse[lb]
    
    del test_iter, test_dataset
    gc.collect()
    
    # print(row)
    return row


import timm
net = timm.create_model('resnest200e', pretrained=0, num_classes=5)
net.load_state_dict(torch.load( '/kaggle/input/UBCModels/UBC-models/resnest200e.in1k_bestAcc0.950_imgsize_1024_onlyRedMaskTileImg_stainmix.pt' ))

net2 = timm.create_model('tf_efficientnetv2_s', pretrained=0, num_classes=5)
net2.load_state_dict(torch.load( '/kaggle/input/UBCModels/UBC-models/tf_efficientnetv2_s.in21k_ft_in1k_swa_imgsize_1024_onlyMaskTileImg_stainmix.pt' )['state_dict'])

net3 = timm.create_model('seresnextaa101d_32x8d', pretrained=0, num_classes=5)
net3.load_state_dict(torch.load( '/kaggle/input/UBCModels/UBC4/seresnextaa101d_32x8d.sw_in12k_ft_in1k_ep16_bestRecall0.735_imgsize_1024_onlyRedMaskTileImg_stainmix.pt' ))

# infer

from tqdm.auto import tqdm
from joblib import Parallel, delayed

net = net.to(try_gpu()).half()
net.eval()
net2 = net2.to(try_gpu()).half()
net2.eval()
net3 = net3.to(try_gpu()).half()
net3.eval()
nets = [net, net2, net3]
max_samples = 24
threshold = 0  # for identifying outliers
submission = []

submission = Parallel(n_jobs=2, backend='loky')(
     delayed(infer_single_image)
     (idx_row, nets=nets, device=try_gpu(), max_samples=max_samples, threshold=threshold)
     for idx_row in tqdm(test_df.iterrows(), total=len(test_df))
 )


# submission

output = pd.DataFrame(submission)[["image_id", "label"]]
output.to_csv("submission.csv", index=False)
print(output)
print("Your submission was successfully saved!")
