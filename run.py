from mmengine import Config

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import os
import math
import argparse

import torch
from torch import nn
import torch.nn.functional as F

from run_utils import get_callbacks, get_time_str, get_opt_lr_sch
from dataset import get_train_data
from model import StyleTransferModel
from cv_common_utils import show_or_save_batch_img_tensor

class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        ########## ================ MODEL ==================== ##############
        self.model = StyleTransferModel(**config.model_config)
        ########## ================ MODEL ==================== ##############
                

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.model.eval()
            with torch.no_grad():
                content_img, style_img = batch['content_img'], batch['style_img']
                b_s = content_img.size(0)
                transfered_img = self.model.transfer(content_img, style_img)
                transfered_img = show_or_save_batch_img_tensor(transfered_img, 
                                                            num_sample_per_row=int(math.sqrt(b_s)), 
                                                            denorm=True,
                                                            mode='return')
                content_img = show_or_save_batch_img_tensor(content_img, 
                                                            num_sample_per_row=int(math.sqrt(b_s)), 
                                                            denorm=True,
                                                            mode='return')
                style_img = show_or_save_batch_img_tensor(style_img,
                                                            num_sample_per_row=int(math.sqrt(b_s)), 
                                                            denorm=True,
                                                            mode='return')
                self.logger.experiment.add_image('transfered_img', 
                                                 transfered_img, 
                                                 global_step=self.global_step,
                                                 dataformats='HWC')
                self.logger.experiment.add_image('content_img',
                                                    content_img,
                                                    global_step=self.global_step,
                                                    dataformats='HWC')
                self.logger.experiment.add_image('style_img',
                                                    style_img,
                                                    global_step=self.global_step,
                                                    dataformats='HWC')
                
            self.model.train()
            
        content_img, style_img = batch['content_img'], batch['style_img']
        loss_dict = self.model.train_loss(content_img, style_img)
        self.log_dict(loss_dict)
        return loss_dict['loss']
    

    def configure_optimizers(self):
        return get_opt_lr_sch(self.config.optimizer_config, 
                              self.config.lr_sche_config,  
                              self.model)
    




def run(args):
    config = Config.fromfile(args.config)
    config = modify_config(config, args)
    
    # make ckp accord to time
    time_str = get_time_str()
    config.ckp_root = '-'.join([time_str, config.ckp_root, f'[{args.run_name}]'])
    config.ckp_config['dirpath'] = config.ckp_root
    os.makedirs(config.ckp_root, exist_ok=True)
    config.run_name = args.run_name
    

    # logger
    logger = None
    if config.logger_type == 'wandb':
        logger = WandbLogger(**config.wandb_config,
                                name=args.run_name)
        logger.log_hyperparams(config)
    elif config.logger_type == 'tb':
        logger = TensorBoardLogger(save_dir=config.ckp_root,
                                name=config.run_name)
    else:
        raise NotImplementedError
    
    # DATA
    print('getting data...')
    train_data, train_loader = get_train_data(config.train_data_config)
    # val_data, val_loader = get_val_data(config.test_data_config)
    print(f'len train_data: {len(train_data)}, len train_loader: {len(train_loader)}.')
    # print(f'len val_data: {len(val_data)}, len val_loader: {len(val_loader)}.')
    print('done.')


    # lr sche 
    if config.lr_sche_config.type in ['linear', 'cosine']:
        if config.lr_sche_config.config.get('warm_up_epoch', None) is not None:
            warm_up_epoch = config.lr_sche_config.config.warm_up_epoch
            config.lr_sche_config.config.pop('warm_up_epoch')
            config.lr_sche_config.config['num_warmup_steps'] = int(warm_up_epoch * len(train_loader))
        else:
            config.lr_sche_config.config['num_warmup_steps'] = 0
        config.lr_sche_config.config['num_training_steps'] = config.num_ep * len(train_loader)
    
    # MODEL
    print('getting model...')
    model = Model(config)
    print(model)
    if 'load_weight_from' in config and config.load_weight_from is not None:
        # only load weights
        # TODO: may add load optimizer state dict
        state_dict = torch.load(config.load_weight_from)['state_dict']
        model.load_state_dict(state_dict)
        print(f'loading weight from {config.load_weight_from}')
    print('done.')
    
    
    callbacks = get_callbacks(config.ckp_config)
    config.dump(os.path.join(config.ckp_root, 'config.py'))
    

    #TRAINING
    print('staring training...')
    if args.find_lr:
        max_steps = args.max_steps
    else:
        max_steps = -1
    resume_ckpt_path = config.resume_ckpt_path if 'resume_ckpt_path' in config else None
    trainer = pl.Trainer(accelerator=config.device,
                         max_epochs=config.num_ep,
                         callbacks=callbacks,
                         logger=logger,
                         enable_progress_bar=True,
                         max_steps=max_steps,
                        #  gradient_clip_val=1.0,
                         **config.trainer_config
                         )
    
    trainer.fit(model,
                train_dataloaders=train_loader,
                # val_dataloaders=val_loader,
                ckpt_path=resume_ckpt_path
                )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="path to mmcv config file")
    parser.add_argument("--run_name", required=True, type=str, help="wandb run name")
    
    parser.add_argument("--find_lr", action='store_true', help="whether to find learning rate")
    parser.add_argument("--max_steps", type=int, default=-100, help='max step to run when find lr')

    # common args to overwrite config
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--wd', type=float, help='Weight decay')
    args = parser.parse_args()
    return args


def modify_config(config, args):
    if args.lr is not None:
        config['optimizer_config']['config']['lr'] = args.lr
    if args.wd is not None:
        config['optimizer_config']['config']['weight_decay'] = args.wd
    return config

if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(42)
    run(args)
