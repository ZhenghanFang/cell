import h5py
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torch import nn
from torch.utils.data import Dataset
from scipy.ndimage import binary_dilation
import pandas as pd
from typing import List, Optional, Tuple
import torchio as tio


def extract_cells_from_xml(xml_path: str) -> list:
    """Extract all cells in a MaMuT xml file.

    Args:
        xml_path: str, path to the xml file.

    Returns:
        spots: list of dict, each dict is the attributes of a cell.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    spots = []
    for spot in root.iter("Spot"):
        spots.append(spot.attrib)
    return spots


def extract_cells_from_xml_voxel(xml_path: str) -> list:
    """Extract all cells in a MaMuT xml file.

    Args:
        xml_path: str, path to the xml file.

    Returns:
        spots: list of dict, each dict includes t, z, y, x of a cell, in voxel index.
    """
    cells = extract_cells_from_xml(xml_path)
    cells = [
        {
            "t": int(float(cell["POSITION_T"])),
            "z": float(cell["POSITION_Z"]),
            "y": float(cell["POSITION_Y"]),
            "x": float(cell["POSITION_X"]),
        }
        for cell in cells
    ]

    voxel_size = [2.0, 0.4200000672000107, 0.4200000672000107]
    voxel_size = np.array(voxel_size) * [1, 4, 4]
    print("voxel size", voxel_size)

    # convert to voxel index
    def convert_to_voxel_index(cells, voxel_size):
        """Convert the coordinates of cells to voxel index."""
        for cell in cells:
            cell["z"] = int(cell["z"] / voxel_size[0])
            cell["y"] = int(cell["y"] / voxel_size[1])
            cell["x"] = int(cell["x"] / voxel_size[2])
        return cells

    cells = convert_to_voxel_index(cells, voxel_size)
    return cells


def extract_patch_from_h5(h5_path: str, t, coor: tuple, size: tuple) -> np.ndarray:
    """Extract an image patch from a HDF5 file.

    Args:
        h5_path: str, path to the HDF5 file.
        t: int, time point.
        coor: tuple, (z, y, x), coordinate of the corner of the patch.
        size: tuple, (size_z, size_y, size_x), size of the patch.

    Returns:
        img: np.ndarray, shape (size_z, size_y, size_x), the extracted image patch.

    Note:
        the h5 file has structure like this:
        t[time point]/s00/[resolution index]/cells
        time point starts from 0
        resolution index is 0, 1, or 2. We extract 0, the highest resolution.
    """

    # open the HDF5 file
    h5_file = h5py.File(h5_path, "r")

    img = h5_file[f"t{t:05d}"]["s00"]["0"]["cells"][
        coor[0] : coor[0] + size[0],
        coor[1] : coor[1] + size[1],
        coor[2] : coor[2] + size[2],
    ]

    return img


def extract_img_from_h5(h5_path: str, resolution: int) -> np.ndarray:
    """Extract images from a HDF5 file.

    Args:
        h5_path: str, path to the HDF5 file.
        resolution: int, resolution index. choose from 0, 1, 2.

    Returns:
        img: np.ndarray, shape (number of time points, depth, height, width).

    Note:
        the h5 file has structure like this:
        t[time point]/s00/[resolution index]/cells
        time point starts from 0
        resolution index is 0, 1, or 2. 0 is full resolution, 1 is downsampled by 2, 2 is downsampled by 4.
    """

    # open the HDF5 file
    h5_file = h5py.File(h5_path, "r")

    # n is the number of time points
    n = len([x for x in h5_file if x.startswith("t")])

    data = []
    # extract images at each time point
    for t in range(n):
        data.append(h5_file[f"t{t:05d}"]["s00"][str(resolution)]["cells"][()])

    return np.array(data)


def create_mask_from_xml(xml_path, img_size):
    cells = extract_cells_from_xml_voxel(xml_path)

    def coor_to_mask(cells, img_size):
        """Convert the coordinates of cells to binary mask."""
        mask = np.zeros(img_size, dtype=np.uint8)
        for cell in cells:
            mask[cell["t"], cell["z"], cell["y"], cell["x"]] = 1
        return mask

    mask = coor_to_mask(cells, img_size)

    # finetune the mask
    new_mask = []
    for i in range(mask.shape[0]):
        mask_ = mask[i].copy()
        mask_ = binary_dilation(mask_, iterations=1)

        mask_uncertain = binary_dilation(mask_, iterations=2)
        mask_uncertain[mask_] = 0

        mask_ = mask_ * 1 + mask_uncertain * (-1)
        new_mask.append(mask_)
    mask = np.array(new_mask)

    return mask


def create_df(patch_size, img_size):
    samples = []
    for t in range(img_size[0]):
        for z in range(0, img_size[1] - patch_size[0], patch_size[0] // 2):
            for y in range(0, img_size[2] - patch_size[1], patch_size[1] // 2):
                for x in range(0, img_size[3] - patch_size[2], patch_size[2] // 2):
                    samples.append({"t": t, "z": z, "y": y, "x": x})
    df = pd.DataFrame(samples)

    splits = []
    for i in range(len(df)):
        if df.loc[i, "t"] < 14:
            split = "train"
        elif df.loc[i, "t"] < 21:
            split = "val"
        else:
            split = "test"
        splits.append(split)
    df["split"] = splits

    return df


class CellDataset(Dataset):
    def __init__(self, image, mask, cells, df, patch_size, split):
        self.image = image
        self.mask = mask
        self.cells = cells
        self.df = df
        self.patch_size = patch_size
        self.split = split

        self.df = self.df[self.df["split"] == split]

        self.training_transform = get_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        t = sample["t"]
        z = sample["z"]
        y = sample["y"]
        x = sample["x"]

        # extract patch
        img_patch = self.image[
            t,
            z : z + self.patch_size[0],
            y : y + self.patch_size[1],
            x : x + self.patch_size[2],
        ]
        label_patch = self.mask[
            t,
            z : z + self.patch_size[0],
            y : y + self.patch_size[1],
            x : x + self.patch_size[2],
        ]

        img_patch = torch.tensor(img_patch[None, :, :, :]).float()
        label_patch = torch.tensor(label_patch[None, :, :, :]).float()

        # data augmentation
        if self.split == "train":
            sample_tio = tio.Subject(
                img=tio.ScalarImage(tensor=img_patch),
                label=tio.LabelMap(tensor=label_patch),
            )
            sample_tio = self.training_transform(sample_tio)
            img_patch, label_patch = sample_tio["img"].data, sample_tio["label"].data

        # extract cells in the patch
        cells_in_patch = np.ones((1000, 3), dtype=int) * -1
        k = 0
        for cell in self.cells:
            if (
                cell["t"] == t
                and cell["z"] >= z
                and cell["z"] < z + self.patch_size[0]
                and cell["y"] >= y
                and cell["y"] < y + self.patch_size[1]
                and cell["x"] >= x
                and cell["x"] < x + self.patch_size[2]
            ):
                cells_in_patch[k] = [cell["z"] - z, cell["y"] - y, cell["x"] - x]
                k += 1

        meta = {"cells": cells_in_patch, "t": t}

        # return
        return img_patch, label_patch, meta


def get_transform():
    training_transform = tio.Compose(
        [
            tio.OneOf(
                {  # either
                    tio.RandomAffine(
                        default_pad_value=0, degrees=40, translation=10
                    ): 0.8,  # random affine
                    # tio.RandomElasticDeformation(): 0.2,  # or random elastic deformation
                },
                p=0.8,
            ),
        ]
    )
    return training_transform


# post processing, get keypoints from output heatmap
def torch_gaussian_blur_3d(x, kernel_size, sigma):
    """Apply Gaussian blur to a 3D Torch tensor.

    Args:
        x (torch.Tensor): 3D image, N, C, D, H, W
        kernel_size (int): kernel size of the Gaussian kernel
        sigma (float): sigma of the Gaussian kernel

    Returns:
        torch.Tensor: 3D image, N, C, D, H, W
    """
    n, c, d, h, w = x.shape

    # create Gaussian kernel
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    x_coor = torch.arange(kernel_size) - (kernel_size - 1) / 2
    grid_x, grid_y, grid_z = torch.meshgrid(x_coor, x_coor, x_coor)
    kernel = torch.exp(-(grid_x**2 + grid_y**2 + grid_z**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Reshape to 2d depthwise convolutional weight
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    # create Gaussian filter
    gaussian_filter = nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        bias=False,
        padding=kernel_size // 2,
        padding_mode="reflect",
    )
    gaussian_filter.weight.data = kernel
    gaussian_filter.weight.requires_grad = False

    # apply Gaussian filter
    gaussian_filter = gaussian_filter.to(x.device)
    x = x.view(n * c, 1, d, h, w)
    x = gaussian_filter(x)
    x = x.view(n, c, d, h, w)

    return x


def get_keypoints_from_heatmap_batch_maxpool_3d(
    heatmap: torch.Tensor,
    max_keypoints: int = 20,
    min_keypoint_pixel_distance: int = 1,
    abs_max_threshold: Optional[float] = None,
    rel_max_threshold: Optional[float] = None,
):
    """Extract keypoints from a batch of 3D heatmaps using maxpooling.

    Modified from the 2D version from:
        - https://github.com/tlpss/keypoint-detection/blob/main/keypoint_detection/utils/heatmap.py

    Args:
        heatmap (torch.Tensor): N, C, D, H, W. heatmap batch
        max_keypoints (int, optional): max number of keypoints to extract, lowering will result in faster execution times. Defaults to 20.
        min_keypoint_pixel_distance (int, optional): _description_. Defaults to 1.

        Following thresholds can be used at inference time to select where you want to be on the AP curve. They should ofc. not be used for training
        abs_max_threshold (Optional[float], optional): _description_. Defaults to None.
        rel_max_threshold (Optional[float], optional): _description_. Defaults to None.

    Returns:
        The extracted keypoints for each batch, channel and heatmap; and their scores
    """

    # TODO: maybe separate the thresholding into another function to make sure it is not used during training, where it should not be used?

    batch_size, n_channels, D, H, W = heatmap.shape

    # smooth heatmap by Gaussian kernel to avoid multiple maxima
    heatmap = torch_gaussian_blur_3d(heatmap, kernel_size=5, sigma=0.5)

    # obtain max_keypoints local maxima for each channel (w/ maxpool)

    kernel = min_keypoint_pixel_distance * 2 + 1
    pad = min_keypoint_pixel_distance
    padded_heatmap = torch.nn.functional.pad(
        heatmap, (pad, pad, pad, pad, pad, pad), mode="constant", value=0.0
    )
    max_pooled_heatmap = torch.nn.functional.max_pool3d(
        padded_heatmap, kernel, stride=1, padding=0
    )
    # if the value equals the original value, it is the local maximum
    local_maxima = max_pooled_heatmap == heatmap
    # all values to zero that are not local maxima
    heatmap = heatmap * local_maxima

    # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
    scores, indices = torch.topk(
        heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True
    )
    # at this point either score > 0.0, in which case the index is a local maximum
    # or score is 0.0, in which case topk returned non-maxima, which will be filtered out later.

    #  remove top-k that are not local maxima and threshold (if required)
    # thresholding shouldn't be done during training

    #  moving them to CPU now to avoid multiple GPU-mem accesses!
    indices = indices.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    # convert indices to 3D coordinates
    indices = np.stack(np.unravel_index(indices, (D, H, W)), axis=-1)
    # filter out points that are below threshold
    filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
    filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
    # determine NMS threshold
    threshold = (
        0.01  # make sure it is > 0 to filter out top-k that are not local maxima
    )
    if abs_max_threshold is not None:
        threshold = max(threshold, abs_max_threshold)
    if rel_max_threshold is not None:
        threshold = max(threshold, rel_max_threshold * heatmap.max())

    # have to do this manually as the number of maxima for each channel can be different
    for batch_idx in range(batch_size):
        for channel_idx in range(n_channels):
            candidates = indices[batch_idx, channel_idx]
            for candidate_idx in range(candidates.shape[0]):

                # these are filtered out directly.
                if scores[batch_idx, channel_idx, candidate_idx] > threshold:
                    # convert to (u,v)
                    filtered_indices[batch_idx][channel_idx].append(
                        candidates[candidate_idx].tolist()
                    )
                    filtered_scores[batch_idx][channel_idx].append(
                        scores[batch_idx, channel_idx, candidate_idx]
                    )

    return filtered_indices, filtered_scores


# def get_keypoints_from_heatmap_batch_maxpool_3d(
#     heatmap: torch.Tensor,
#     max_keypoints: int = 20,
#     min_keypoint_pixel_distance: int = 1,
#     abs_max_threshold: Optional[float] = None,
#     rel_max_threshold: Optional[float] = None,
# ):
#     """Extract keypoints from a batch of 3D heatmaps using maxpooling.

#     Modified from the 2D version from:
#         - https://github.com/tlpss/keypoint-detection/blob/main/keypoint_detection/utils/heatmap.py

#     Args:
#         heatmap (torch.Tensor): N, C, D, H, W. heatmap batch
#         max_keypoints (int, optional): max number of keypoints to extract, lowering will result in faster execution times. Defaults to 20.
#         min_keypoint_pixel_distance (int, optional): _description_. Defaults to 1.

#         Following thresholds can be used at inference time to select where you want to be on the AP curve. They should ofc. not be used for training
#         abs_max_threshold (Optional[float], optional): _description_. Defaults to None.
#         rel_max_threshold (Optional[float], optional): _description_. Defaults to None.

#     Returns:
#         The extracted keypoints for each batch, channel and heatmap; and their scores
#     """

#     # TODO: maybe separate the thresholding into another function to make sure it is not used during training, where it should not be used?

#     batch_size, n_channels, D, H, W = heatmap.shape

#     # obtain max_keypoints local maxima for each channel (w/ maxpool)

#     kernel = min_keypoint_pixel_distance * 2 + 1
#     pad = min_keypoint_pixel_distance
#     # exclude border keypoints by padding with highest possible value
#     # bc the borders are more susceptible to noise and could result in false positives
#     padded_heatmap = torch.nn.functional.pad(
#         heatmap, (pad, pad, pad, pad, pad, pad), mode="constant", value=1.0
#     )
#     max_pooled_heatmap = torch.nn.functional.max_pool3d(
#         padded_heatmap, kernel, stride=1, padding=0
#     )
#     # if the value equals the original value, it is the local maximum
#     local_maxima = max_pooled_heatmap == heatmap
#     # all values to zero that are not local maxima
#     heatmap = heatmap * local_maxima

#     # determine NMS threshold
#     threshold = (
#         0.01  # make sure it is > 0 to filter out top-k that are not local maxima
#     )
#     if abs_max_threshold is not None:
#         threshold = max(threshold, abs_max_threshold)
#     if rel_max_threshold is not None:
#         threshold = max(threshold, rel_max_threshold * heatmap.max())

#     #  find top-k keypoints, exclude points that have scores below threshold and those that are too close to each other
#     filtered_scores_all = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
#     filtered_coors_all = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
#     for batch_idx in range(batch_size):
#         for channel_idx in range(n_channels):
#             heat = heatmap[batch_idx, channel_idx].view(-1)
#             scores, indices = torch.sort(heat, descending=True)
#             scores = scores.detach().cpu().numpy()
#             indices = indices.detach().cpu().numpy()
#             filtered_scores = []
#             filtered_coors = []
#             for i in range(len(scores)):
#                 coor = np.unravel_index(indices[i], (D, H, W))
#                 if len(filtered_scores) >= max_keypoints or scores[i] < threshold:
#                     break

#                 # do not add if too close to an already added keypoint
#                 if len(filtered_coors) > 0:
#                     dist = np.linalg.norm(
#                         np.array(coor)[None, :] - np.array(filtered_coors), axis=1
#                     )
#                     if np.min(dist) <= min_keypoint_pixel_distance:
#                         continue

#                 filtered_scores.append(scores[i])
#                 filtered_coors.append(coor)

#             filtered_scores_all[batch_idx][channel_idx] = filtered_scores
#             filtered_coors_all[batch_idx][channel_idx] = filtered_coors

#     return filtered_coors_all, filtered_scores_all


def keypoint_distance(gt, pred):
    """
    Args:
        gt: ground truth keypoints, shape (n_gt, 3)
        pred: predicted keypoints, shape (n_pred, 3)
    """
    gt = np.array(gt)
    pred = np.array(pred)
    distances = np.sqrt(((gt[:, None, :] - pred[None, :, :]) ** 2).sum(axis=-1))
    return distances


def match_keypoints_3d(pred, gt, pixel_distance_threshold=5):
    """
    Args:
        pred: predicted keypoints, shape (n_pred, 3)
        gt: ground truth keypoints, shape (n_gt, 3)
        pixel_distance_threshold: threshold for matching keypoints, float

    Returns:
        matches: list of list, matches[i] is a list of indices of matched gt keypoints for the i-th predicted keypoint
    """
    dist_mtx = keypoint_distance(gt, pred)

    # match keypoints
    matches = []
    for i in range(len(pred)):
        matches.append([])
        for j, dist in enumerate(dist_mtx[:, i]):
            if dist < pixel_distance_threshold:
                matches[-1].append(j)

    return matches


def calculate_precision_recall_curve(pred_keypoints, n_gt_keypoints):
    """Calculate precision-recall curve from predicted keypoints and ground truth keypoints

    Args:
        pred_keypoints (List[Dict]): List of predicted keypoints, each dict should be in the following format:
            {
                "img_id": int,
                "coor": Tuple[int, int, int],
                "score": float,
                "matched": List[Tuple[int, Tuple[int, int, int]]], [(gt_img_id, gt_coor), ...]
            }
        n_gt_keypoints (int): Number of ground truth keypoints

    Returns:
        prec: list of precision values
        rec: list of recall values
    """
    pred_keypoints = sorted(pred_keypoints, key=lambda x: x["score"], reverse=True)
    tp, fp = 0, 0
    matched_gt_keypoints = set()
    prec, rec = [], []
    for pred_keypoint in pred_keypoints:
        if len(pred_keypoint["matched"]) == 0:
            fp += 1
        else:
            tp += 1
            for matched in pred_keypoint["matched"]:
                matched_gt_keypoints.add(matched)

        prec.append(tp / (tp + fp))
        rec.append(len(matched_gt_keypoints) / n_gt_keypoints)

    return prec, rec


def calculate_precision_recall_f1_at_threshold(
    pred_keypoints, n_gt_keypoints, threshold
):
    """Calculate precision, recall, f1 at a given threshold

    Args:
        pred_keypoints (List[Dict]): List of predicted keypoints, each dict should be in the following format:
            {
                "img_id": int,
                "coor": Tuple[int, int, int],
                "score": float,
                "matched": List[Tuple[int, Tuple[int, int, int]]], [(gt_img_id, gt_coor), ...]
            }
        n_gt_keypoints (int): Number of ground truth keypoints

    Returns:
        precision, recall, f1
    """
    pred_keypoints = sorted(pred_keypoints, key=lambda x: x["score"], reverse=True)
    if pred_keypoints[-1]["score"] > threshold:
        print(
            "Warning: threshold is lower than the lowest score, ",
            pred_keypoints[-1]["score"],
        )
    pred_keypoints = [x for x in pred_keypoints if x["score"] > threshold]
    tp, fp = 0, 0
    matched_gt_keypoints = set()
    for pred_keypoint in pred_keypoints:
        if len(pred_keypoint["matched"]) == 0:
            fp += 1
        else:
            tp += 1
            for matched in pred_keypoint["matched"]:
                matched_gt_keypoints.add(matched)

    prec = tp / (tp + fp)
    rec = len(matched_gt_keypoints) / n_gt_keypoints
    f1 = 2 * (prec * rec) / (prec + rec)

    return prec, rec, f1


def calculate_average_precision(prec: list, rec: list) -> float:
    """Calculate average precision from precision recall curve.

    Args:
        prec: List of precision values.
        rec: List of recall values, same length as prec.

    Returns:
        float, Average precision.
    """
    ap = 0.0
    for i in range(1, len(rec)):
        ap += (rec[i] - rec[i - 1]) * prec[i]

    return ap