# PyTorch lightning version

import argparse
import datetime
import os
import signal
import sys
import time
import shutil

import numpy as np
import timm
import torch
import torch.nn as nn
from fastai.vision.all import URLs, untar_data
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder

from dime import CMIEstimator, MaskingPretrainer
from dime.resnet_imagenet import Predictor, ResNet18Backbone, ValueNetwork
from dime.utils import MaskLayer2d
from dime.vit import PredictorViT, ValueNetworkViT

vit_model_options = ['vit_small_patch16_224', 'vit_tiny_patch16_224']
resnet_model_options = ['resnet18']

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mask_type', type=str,
                    default='zero',
                    choices=['gaussian', 'zero'],
                    help="Type of mask to apply: either Gaussain blur (gaussian) or zero-out (zero)")
parser.add_argument('--mask_width', type=int,
                    default=14,
                    choices=[7, 14],
                    help="Mask width to use in the mask layer")
parser.add_argument('--lr', type=float,
                    default=1e-5,
                    help="Learning rate used train the network")
parser.add_argument('--backbone', type=str,
                    default='vit',
                    choices=['vit', 'resnet'],
                    help="Backbone used to train the network")
parser.add_argument('--pretrained_model_name', type=str,
                    default='vit_small_patch16_224',
                    choices=vit_model_options+resnet_model_options,
                    help="Name of the pretrained model to use")
parser.add_argument('--trial', type=int,
                    default=1,
                    help="Trial Number")
parser.add_argument('--load_pretrain_predictor', type=str,
                    default=None,
                    help="Path to a saved pretraining predictor state_dict to skip pretraining.")

if __name__ == '__main__':
    def timestamped_log(msg: str) -> None:
        """Print to stdout and append to the per-run log file."""
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_file_path, "a") as lf:
            lf.write(line + "\n")

    def _handle_signal(sig, frame):
        try:
            sig_name = signal.Signals(sig).name  # type: ignore[arg-type]
        except Exception:
            sig_name = str(sig)
        timestamped_log(f"Received signal {sig_name}; attempting graceful shutdown.")
        sys.exit(0)

    acc_metric = Accuracy(task='multiclass', num_classes=10)

    # Parse args
    args = parser.parse_args()
    mask_type = args.mask_type
    image_size = 224
    mask_width = args.mask_width
    network_type = args.backbone
    pretrained_model_name = args.pretrained_model_name
    lr = args.lr
    if lr == 1e-3:
        min_lr = 1e-6
    else:
        min_lr = 1e-8

    if (network_type == 'vit' and pretrained_model_name not in vit_model_options) \
       or (network_type == 'resnet' and pretrained_model_name not in resnet_model_options):
        raise argparse.ArgumentError("Network type and model name are not compatible")

    mask_layer = MaskLayer2d(append=False, mask_width=mask_width, patch_size=image_size/mask_width)

    run_description = f"max_features_50_{pretrained_model_name}_lr_1e-5_use_entropy_True_{mask_type}_mask_width_{mask_width}_trial_{args.trial}"
    run_dir = os.path.join("results", run_description)
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, "run_log.txt")

    # Register signal handlers early so a system kill is recorded before exit.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _handle_signal)
    timestamped_log(f"Starting run {run_description}")
    timestamped_log(f"Args: gpu={args.gpu}, lr={lr}, backbone={network_type}, model={pretrained_model_name}, mask_width={mask_width}, mask_type={mask_type}")
        
    device = torch.device('cuda', args.gpu)
    dataset_path = "/homes/gws/<username>/.fastai/data/imagenette2-320"
    if not os.path.exists(dataset_path):
        dataset_path = str(untar_data(URLs.IMAGENETTE_320))
        
    norm_constants = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # Setup for data loading.
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*norm_constants),
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*norm_constants),
    ])

    # Get Datasets
    train_dataset_train_transforms = ImageFolder(dataset_path+'/train', transforms_train)
    train_dataset_all_len = len(train_dataset_train_transforms)
    timestamped_log(f"Loaded dataset from {dataset_path}; train size={train_dataset_all_len}")

    # Get train and val indices
    np.random.seed(0)
    val_inds = np.sort(np.random.choice(train_dataset_all_len, size=int(train_dataset_all_len*0.1), replace=False))
    train_inds = np.setdiff1d(np.arange(train_dataset_all_len), val_inds)

    train_dataset = torch.utils.data.Subset(train_dataset_train_transforms, train_inds)

    train_dataset_test_transforms = ImageFolder(dataset_path+'/train', transforms_test)
    val_dataset = torch.utils.data.Subset(train_dataset_test_transforms, val_inds)

    test_dataset = ImageFolder(dataset_path+'/val', transforms_test)

    # Prepare dataloaders.
    mbsize = 32
    train_dataloader = DataLoader(train_dataset, batch_size=mbsize, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=mbsize, pin_memory=True, num_workers=2)

    d_in = image_size * image_size
    d_out = 10
    
    # Make results directory.
    if not os.path.exists('results'):
        os.makedirs('results')
    
    if network_type == 'vit':
        backbone = timm.create_model(pretrained_model_name, pretrained=True)
        predictor = PredictorViT(backbone)
        value_network = ValueNetworkViT(backbone)
    else:
        # Set up networks.
        backbone, expansion = ResNet18Backbone(eval(pretrained_model_name + '(pretrained=True)'))
        predictor = Predictor(backbone, expansion)
        block_layer_stride = 1
        if mask_width == 14:
            block_layer_stride = 0.5
        
        value_network = ValueNetwork(backbone, expansion, block_layer_stride=block_layer_stride)

    starting_time = time.time()
    pretrain_predictor_path = os.path.join(run_dir, "pretrain_predictor_state_dict.pt")
    if args.load_pretrain_predictor:
        timestamped_log(f"Loading pretrain predictor weights from {args.load_pretrain_predictor}")
        state = torch.load(args.load_pretrain_predictor, map_location="cpu")
        predictor.load_state_dict(state)
        timestamped_log("Skipped mask pretraining; loaded provided weights.")
    else:
        timestamped_log("Beginning mask pretraining phase.")
        pretrain = MaskingPretrainer(
                predictor,
                mask_layer,
                lr=1e-5,
                loss_fn=nn.CrossEntropyLoss(),
                val_loss_fn=acc_metric)
        
        trainer = Trainer(
                accelerator='gpu',
                devices=[args.gpu],
                max_epochs=50,
                num_sanity_val_steps=0
            )
        trainer.fit(pretrain, train_dataloader, val_dataloader)
        timestamped_log("Finished mask pretraining.")
        torch.save(predictor.state_dict(), pretrain_predictor_path)
        timestamped_log(f"Saved pretrain predictor state_dict to {pretrain_predictor_path}")

    logger = TensorBoardLogger("logs", name=f"{run_description}", default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(
                save_top_k=1,
                monitor='Perf Val/Final',
                mode='max',
                filename='best_val_perfomance_model',
                verbose=False
            )
    
    # Jointly train predictor and value networks
    greedy_cmi_estimator = CMIEstimator(value_network,
                                        predictor,
                                        mask_layer,
                                        lr=lr,
                                        min_lr=min_lr,
                                        max_features=50,
                                        eps=0.05,
                                        loss_fn=nn.CrossEntropyLoss(reduction='none'),
                                        val_loss_fn=Accuracy(task='multiclass', num_classes=10),
                                        eps_decay=0.2,
                                        eps_steps=10,
                                        patience=3,
                                        feature_costs=None)
    
    trainer = Trainer(
                accelerator='gpu',
                devices=[args.gpu],
                max_epochs=250,
                precision=16,
                logger=logger,
                num_sanity_val_steps=0,
                callbacks=[checkpoint_callback]
            )

    try:
        trainer.fit(greedy_cmi_estimator, train_dataloader, val_dataloader)
    except Exception as e:
        timestamped_log(f"Training failed with exception: {e}")
        raise
    finally:
        training_time = time.time() - starting_time
        timestamped_log(f"Training time (s): {training_time:.2f}")
        with open(os.path.join(run_dir, "training_time.txt"), 'a') as f:
            f.write(f"Training time = {training_time}\n")

    # Persist checkpoints for later reuse.
    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        dest_best = os.path.join(run_dir, "best.ckpt")
        shutil.copy(best_ckpt_path, dest_best)
        timestamped_log(f"Saved best checkpoint to {dest_best}")
    else:
        timestamped_log("No best checkpoint was produced.")

    last_ckpt_path = os.path.join(run_dir, "last.ckpt")
    trainer.save_checkpoint(last_ckpt_path)
    timestamped_log(f"Saved final checkpoint to {last_ckpt_path}")

    # Save lightweight weights for downstream use without Lightning.
    torch.save(greedy_cmi_estimator.predictor.state_dict(), os.path.join(run_dir, "predictor_state_dict.pt"))
    torch.save(greedy_cmi_estimator.value_network.state_dict(), os.path.join(run_dir, "value_network_state_dict.pt"))
    timestamped_log("Exported predictor and value network state dicts.")
