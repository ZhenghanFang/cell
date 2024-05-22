import h5py
import numpy as np
import xml.etree.ElementTree as ET
from scipy.ndimage import binary_dilation
import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm

from utils import (
    CellDataset,
    extract_cells_from_xml,
    extract_patch_from_h5,
    extract_img_from_h5,
    create_mask_from_xml,
    create_df,
    extract_cells_from_xml_voxel,
)
from unet3d.models import ResidualUNet3D


h5_path = "data/A4_6mo_no_dextran_long_one/A4_wt_pdcre_rce_6mo_nod_R2_ma.h5.h5"
img = extract_img_from_h5(h5_path, resolution=2)

xml_path = "data/A4_6mo_no_dextran_long_one/A4_wt_pdcre_rce_6mo_nod_R2_ma-mamut.xml"
mask = create_mask_from_xml(xml_path, img.shape)
cells = extract_cells_from_xml_voxel(xml_path)

patch_size = (64, 64, 64)
df = create_df(patch_size, img.shape)

train_dataset = CellDataset(img, mask, cells, df, patch_size, "train")
val_dataset = CellDataset(img, mask, cells, df, patch_size, "val")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
dataloaders = {"train": train_loader, "val": val_loader}

checkpoint_path = "checkpoint.pth"
device = "cuda"


def dice_loss(pred, target):
    """
    Args:
        pred: tensor, shape (batch_size, 1, depth, height, width)
        target: tensor, shape (batch_size, 1, depth, height, width)
            target == -1 means the region is uncertain and excluded from loss calculation.

    """
    smooth = 1.0
    excluded = (target == -1).float()
    pred = pred * (1 - excluded)
    target = target * (1 - excluded)

    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=25):
    print("len train loader", len(dataloaders["train"]))
    print("len val loader", len(dataloaders["val"]))

    best_loss = 1e10

    for epoch in tqdm(range(num_epochs)):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for iter, (inputs, labels, _) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    outputs = F.sigmoid(outputs)
                    loss = dice_loss(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
                metrics["loss"] += loss.item() * inputs.size(0)

                if iter % 10 == 0:
                    print(
                        "{} Iteration {}/{} Loss: {:.4f}".format(
                            phase, iter, len(dataloaders[phase]), loss.item()
                        )
                    )

            epoch_loss = metrics["loss"] / epoch_samples
            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            # if phase == 'train':
            #   scheduler.step()
            #   for param_group in optimizer.param_groups:
            #       print("LR", param_group['lr'])

            # save the model weights
            if phase == "val" and epoch_loss < best_loss:
                print(f"saving best model to {checkpoint_path}")
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path)

    print("Best val loss: {:4f}".format(best_loss))


model = ResidualUNet3D(in_channels=1, out_channels=1)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = None


train_model(model, optimizer, scheduler, num_epochs=500)
