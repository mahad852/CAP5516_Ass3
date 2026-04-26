import os
import torch
from typing import Optional
from skimage import filters, measure, morphology
from scipy import ndimage as ndi
import numpy as np
import random

def get_all_image_ids(root: str):
    im_ids = []

    for organ_dir in os.listdir(root):
        if organ_dir.startswith(".") or organ_dir.startswith("_"):
            continue
        
        image_dir = os.path.join(root, organ_dir, "tissue images")
        for image_fname in os.listdir(image_dir):
            img_id = image_fname.split(".")[0]
            im_ids.append(img_id)
    
    return im_ids


def sample_one_point_from_binary_mask(mask_hw):
    ys, xs = torch.where(mask_hw > 0)
    idx = torch.randint(0, ys.numel(), (1,), device=mask_hw.device)
    return torch.stack([xs[idx], ys[idx]], dim=-1).float().squeeze(0)  # [2]

def box_from_mask(
    binary_mask_bhw: torch.Tensor,
    padding: int = 2,
) -> torch.Tensor:
    B, H, W = binary_mask_bhw.shape

    boxes = []

    for b in range(B):
        mask = binary_mask_bhw[b]
        ys, xs = torch.where(mask > 0)

        x1 = max(xs.min().item() - padding, 0)
        x2 = min(xs.max().item() + padding, W - 1)
        y1 = max(ys.min().item() - padding, 0)
        y2 = min(ys.max().item() + padding, H - 1)

        box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=mask.device)
        boxes.append(box)

    boxes = torch.stack(boxes, dim=0)  # (B, 4)
    boxes = boxes.unsqueeze(1)         # (B, 1, 4)

    return boxes

def get_grid_points(image_size=512, points_per_side=32):
    x = torch.linspace(0, image_size - 1, points_per_side)
    y = torch.linspace(0, image_size - 1, points_per_side)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

def sample_one_point_per_instance(gt_label_np):
    points = []

    ids = np.unique(gt_label_np)
    ids = ids[ids != 0]

    for inst_id in ids:
        ys, xs = np.where(gt_label_np == inst_id)
        j = random.randint(a=0, b=len(xs) - 1)
        points.append([xs[j], ys[j]])

    if len(points) == 0:
        return torch.empty((0, 2), dtype=torch.float32)

    return torch.tensor(points, dtype=torch.float32)

def generate_proposal_boxes_from_image(
    image_np: np.ndarray,
    min_area: int = 20,
    max_area: int = 3000,
    padding: int = 4,
):

    if image_np.max() <= 1:
        image_np = (image_np * 255).astype(np.uint8)

    gray = image_np.mean(axis=-1)
    inv = 255 - gray

    thresh = filters.threshold_otsu(inv)
    binary = inv > thresh

    binary = morphology.remove_small_objects(binary, min_size=min_area)
    binary = morphology.opening(binary, morphology.disk(1))
    binary = ndi.binary_fill_holes(binary)

    labeled = measure.label(binary)
    props = measure.regionprops(labeled)

    H, W = binary.shape
    boxes = []

    for p in props:
        area = p.area
        if area < min_area or area > max_area:
            continue

        y1, x1, y2, x2 = p.bbox

        x1 = max(x1 - padding, 0)
        y1 = max(y1 - padding, 0)
        x2 = min(x2 + padding, W - 1)
        y2 = min(y2 + padding, H - 1)

        boxes.append([x1, y1, x2, y2])

    if len(boxes) == 0:
        return torch.empty((0, 4), dtype=torch.float32)

    return torch.tensor(boxes, dtype=torch.float32)

def binary_dice(pred_label: np.ndarray, gt_label: np.ndarray, eps: float = 1e-7):
    pred = pred_label > 0
    gt = gt_label > 0

    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()

    return (2 * inter + eps) / (denom + eps)


def compute_iou_matrix(gt_label: np.ndarray, pred_label: np.ndarray):
    gt_ids = np.array([i for i in np.unique(gt_label) if i != 0])
    pred_ids = np.array([i for i in np.unique(pred_label) if i != 0])

    iou = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)

    for gi, gid in enumerate(gt_ids):
        g = gt_label == gid
        for pi, pid in enumerate(pred_ids):
            p = pred_label == pid
            inter = np.logical_and(g, p).sum()
            union = np.logical_or(g, p).sum()
            if union > 0:
                iou[gi, pi] = inter / union

    return gt_ids, pred_ids, iou

def mask_iou(a, b, eps=1e-6):
    inter = torch.logical_and(a, b).sum().float()
    union = torch.logical_or(a, b).sum().float()
    return inter / (union + eps)


def aji_score(gt_label: np.ndarray, pred_label: np.ndarray, eps: float = 1e-7):
    gt_ids, pred_ids, iou = compute_iou_matrix(gt_label, pred_label)

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    used_pred = set()
    intersection_sum = 0
    union_sum = 0

    for gi, gid in enumerate(gt_ids):
        best_pi = np.argmax(iou[gi])
        best_iou = iou[gi, best_pi]

        g = gt_label == gid

        if best_iou > 0:
            pid = pred_ids[best_pi]
            p = pred_label == pid

            intersection_sum += np.logical_and(g, p).sum()
            union_sum += np.logical_or(g, p).sum()
            used_pred.add(pid)
        else:
            union_sum += g.sum()

    for pid in pred_ids:
        if pid not in used_pred:
            union_sum += (pred_label == pid).sum()

    return (intersection_sum + eps) / (union_sum + eps)


def pq_score(gt_label: np.ndarray, pred_label: np.ndarray, iou_threshold: float = 0.5, eps: float = 1e-7):
    gt_ids, pred_ids, iou = compute_iou_matrix(gt_label, pred_label)

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0

    matches = []
    used_gt = set()
    used_pred = set()

    pairs = np.argwhere(iou > iou_threshold)
    pairs = sorted(pairs, key=lambda x: iou[x[0], x[1]], reverse=True)

    for gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)
        matches.append(iou[gi, pi])

    tp = len(matches)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp

    if tp == 0:
        return 0.0

    sq = np.sum(matches) / (tp + eps)
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + eps)

    return sq * rq