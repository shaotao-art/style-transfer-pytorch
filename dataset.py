from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tt

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from functools import partial
import os
from PIL import Image
import random

class StyleTransferDataset(Dataset):
    def __init__(self, content_img_root, style_img_root):
        self.content_img_lst = [f'{content_img_root}/{i}' for i in os.listdir(content_img_root)]
        self.style_img_lst = [f'{style_img_root}/{i}' for i in os.listdir(style_img_root)]
        self.content_img_preprocess = tt.Compose([
            tt.Resize(384),
            tt.RandomCrop((256, 256)),
            tt.ToTensor(),
            tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.style_img_preprocess = tt.Compose([
            tt.Resize(512),
            tt.RandomCrop((256, 256)),
            tt.ToTensor(),
            tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __len__(self):
        return 4000

    def __getitem__(self, idx):
        content_img_idx = random.randint(0, len(self.content_img_lst) - 1)
        style_img_idx = random.randint(0, len(self.style_img_lst) - 1)
        content_img = self.content_img_preprocess(Image.open(self.content_img_lst[content_img_idx]).convert('RGB'))
        style_img = self.style_img_preprocess(Image.open(self.style_img_lst[style_img_idx]).convert('RGB'))
        return dict(content_img=content_img, style_img=style_img)


def get_train_data(data_config):
    dataset = StyleTransferDataset(**data_config.dataset_config)
    data_loader = DataLoader(dataset=dataset, 
                             **data_config.data_loader_config, 
                             shuffle=True)
    return dataset, data_loader
    