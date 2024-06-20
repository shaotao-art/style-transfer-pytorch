import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from copy import deepcopy
import random
from typing import List
from functools import partial
import torch
from torchvision.utils import make_grid
from typing import Dict, List, Tuple



def array2tensor(inp: np.ndarray):
    assert isinstance(inp, np.ndarray)
    return torch.tensor(inp)

def print_lst_tensor_shape(lst: List, name=None):
    if name is not None:
        print(f'{name} is a list of tensor with shape')
    for x in lst:
        print('\t', x.shape)


def append_dims(inp: torch.Tensor, target_len: int):
    """unsqueeze tensor at dim=-1, until ndim=target_len"""
    assert len(inp.shape) < target_len
    while len(inp.shape) < target_len:
        inp = inp.unsqueeze(-1)
    return inp

def denorm_img(inp: torch.Tensor,
               mean=torch.tensor([0.485, 0.456, 0.406]), 
               std=torch.tensor([0.229, 0.224, 0.225])):
    """denorm tensor of shape (b[optional], c, h, w) 
    """
    assert len(inp.shape) == 3 or len(inp.shape) == 4
    mean, std = list(map(partial(append_dims, target_len=3), [mean, std]))
    return inp * std + mean




def batch_img_tensor_to_img_lst(inp: torch.Tensor) -> List[np.ndarray]:
    """convert tensor of shape (b, 3, h, w)
    to a list of np array of shape (h, w, 3)

    Args:
        inp (torch.Tensor): _description_

    Returns:
        List[np.ndarray]: _description_
    """
    assert len(inp.shape) == 4 and isinstance(inp, torch.Tensor) and (inp.dtype == torch.float32)
    denormed_inp = denorm_img(inp)
    out = []
    for i in range(denormed_inp.shape[0]):
        out.append((denormed_inp[i].permute(1, 2, 0).contiguous().numpy() * 255).astype(np.uint8))
    return out



def show_img(img: np.ndarray):
    print(f'showing img with shape: {img.shape}')
    plt.tight_layout()
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def dprint(debug: bool, *args):
    if debug == True:
        print(args)
    else:
        pass

def show_or_save_batch_img_tensor(img_tensor: torch.Tensor, 
                                  num_sample_per_row: int, 
                                  denorm: bool = True, 
                                  mode: str = 'show', 
                                  save_p: str = None,
                                  norm_type: str = 'imagenet'):
    assert mode in ['show', 'save', 'all', 'return']
    if img_tensor.device != torch.device('cpu'):
        img_tensor = img_tensor.cpu()
    if denorm:
        assert norm_type in ['imagenet', '0.5']
        if norm_type == 'imagenet':
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            img_tensor = img_tensor * std[:, None, None] + mean[:, None, None]
            img_tensor = torch.clip(img_tensor, 0.0, 1.0)
        elif norm_type == '0.5':
            img_tensor = torch.clip((img_tensor + 1.0) / 2.0, 0.0, 1.0)
        else:
            raise NotImplementedError
        
    img = make_grid(img_tensor, nrow=num_sample_per_row)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    if mode == 'show':
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    if mode == 'save':
        assert save_p is not None
        img = Image.fromarray(img)
        img.save(save_p)
        print(f'saving sample img to {save_p}')
    if mode == 'all':
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        assert save_p is not None
        img = Image.fromarray(img)
        img.save(save_p)
        print(f'saving sample img to {save_p}')
    if mode == 'return':
        return img

           

def print_model_num_params_and_size(model):
    MB = 1024 * 1024
    cal_num_parameters = lambda module: sum([p.numel() for p in module.parameters() if p.requires_grad == True])
    num_param_to_MB = lambda num_parameters: num_parameters * 4  / MB
    total_num_params = cal_num_parameters(model) 
    print(f'model #params: {total_num_params / (10 ** 6)}M, fp32 model size: {num_param_to_MB(total_num_params)} MB') 
    