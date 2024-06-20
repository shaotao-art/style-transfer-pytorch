device = 'cuda'

num_ep = 200
optimizer_config = dict(
    type='adamw',
    config = dict(
        lr = 1e-4,
        # momentum=0.9,
    )
)

lr_sche_config = dict(
    type = 'constant',
    config = dict(
        # epoches=[60, 80],
        # muls=[0.1, 0.1]
    )
)

model_config = dict(
    use_relu=True, 
    style_loss_w=10.0
)


                
cifar_data_root = 'DATA'
train_data_config = dict(
    dataset_config = dict(
        content_img_root='/home/dmt/aha/data/clip-for-segmentation-datasets/VOCdevkit/VOC2012/JPEGImages',
        style_img_root='/home/dmt/shao-tao-working-dir/style-transfer-main/train_2',
    ), 

    data_loader_config = dict(
        batch_size = 8,
        num_workers = 4,
    )
)



resume_ckpt_path = None
load_weight_from = None

# ckp
ckp_config = dict(
   save_last=None, 
   every_n_epochs=None,
#    monitor='val_mae',
#    mode='min',
#    filename='{epoch}-{val_mae:.3f}'
)

# trainer config
trainer_config = dict(
    log_every_n_steps=5,
    precision='32',
    # val_check_interval=0.5, # val after k training batch 0.0-1.0, or a int
    check_val_every_n_epoch=1
)


# LOGGING
logger_type = 'tb'
wandb_config = dict(
    project = 'nlp-cls',
    offline = True
)
ckp_root = f'[{wandb_config["project"]}]'
