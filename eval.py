import argparse
import os
import torch
import cv2
import numpy as np
from NucleiDataset import NucleiDataset
from utils import get_all_image_ids, box_from_mask
import random
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.optim as Optim
from torch.utils.data import DataLoader, Subset

from transformers import SamModel, SamProcessor
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, get_peft_model_state_dict
from monai.losses import DiceLoss
from utils import pq_score, binary_dice, aji_score, compute_iou_matrix, mask_iou, sample_one_point_per_instance,  sample_one_point_from_binary_mask, get_grid_points, generate_proposal_boxes_from_image
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=FutureWarning)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sam_path", default=None, type=str)
    parser.add_argument("--lora_path", required=True, type=str)
    parser.add_argument("--approach", default="point", choices=["point", "auto"])
    parser.add_argument("--fold", default=0, choices=[0, 1, 2, 3, 4], type=int)

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_folds(img_ids: list[str], num_folds=5):
    random.shuffle(img_ids)
    fold_len = len(img_ids) // num_folds    
    folds_img_ids = []

    for fold_idx in range(num_folds):
        start_idx = fold_idx * fold_len
        folds_img_ids.append(img_ids[start_idx:start_idx+fold_len])

    return folds_img_ids

def get_train_val_img_ids(fold_img_ids, val_idx):
    train_img_ids = []
    val_img_ids = fold_img_ids[val_idx]

    for fold_idx, fold in enumerate(fold_img_ids):
        if fold_idx != val_idx:
            train_img_ids.extend(fold)
    
    return train_img_ids, val_img_ids

class MaskResizeOnly:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, mask_pil):
        mask_pil = TF.resize(mask_pil, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        mask_np = np.array(mask_pil, dtype=np.int64)
        return torch.from_numpy(mask_np).long()


def get_mask_transforms():
    return transforms.Compose([
        MaskResizeOnly(size=(256, 256))
    ])

def get_model(sam_path: str = None) -> PeftModel:
    model_name = "facebook/sam-vit-base" if sam_path is None else sam_path
    model = SamModel.from_pretrained(model_name)
    processor = SamProcessor.from_pretrained(model_name)

    return model, processor

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    

@torch.no_grad()
def predict_instances_with_points(
    model,
    pixel_values,
    points,
    device,
    orig_size=512,
    target_size=1024,
    out_size=256,
    mask_threshold=0.0,
    score_threshold=0.88,
    min_area=8,
    max_area=2500,
    nms_iou=0.5,
):
    model.eval()

    scale = target_size / orig_size
    scaled_points = points * scale

    input_points = scaled_points.unsqueeze(1).unsqueeze(0).to(device)  # [1,N,1,2]
    input_labels = torch.ones(input_points.shape[:3], dtype=torch.long, device=device)

    outputs = model(
        pixel_values=pixel_values.to(device),
        input_points=input_points,
        input_labels=input_labels,
        multimask_output=False,
    )

    masks = outputs.pred_masks[0, :, 0]  # [N,256,256]
    scores = outputs.iou_scores[0, :, 0] # [N]

    order = torch.argsort(scores, descending=True)

    kept_masks = []

    for idx in order:
        if scores[idx] < score_threshold:
            continue

        m = masks[idx] > mask_threshold
        area = m.sum().item()

        if area < min_area or area > max_area:
            continue

        duplicate = False
        for km in kept_masks:
            if mask_iou(m, km) > nms_iou:
                duplicate = True
                break

        if duplicate:
            continue

        kept_masks.append(m)

    pred_label = np.zeros((out_size, out_size), dtype=np.int32)

    for i, m in enumerate(kept_masks):
        m_np = m.detach().cpu().numpy().astype(bool)
        m_np = np.logical_and(m_np, pred_label == 0)

        if m_np.sum() == 0:
            continue

        pred_label[m_np] = i + 1

    return pred_label

def _matched_color_maps(gt_mask, pred_mask, iou_threshold=0.1, background_id=0):
    gt_ids, pred_ids, ious = compute_iou_matrix(gt_label=gt_mask, pred_label=pred_mask)

    n_colors = len(gt_ids) + len(pred_ids) + 1

    rng = np.random.default_rng(0)
    base = plt.cm.tab20(np.linspace(0, 1, min(n_colors, 20)))
    extra = rng.random((max(0, n_colors - 20), 4))
    extra[:, 3] = 1.0
    colors = np.vstack([base, extra]) if len(extra) else base

    gt_rgb = np.zeros((*gt_mask.shape, 4), dtype=float)
    pred_rgb = np.zeros((*pred_mask.shape, 4), dtype=float)

    color_idx = 0
    matched_gt = set()
    matched_pred = set()

    if len(gt_ids) and len(pred_ids):
        row_ind, col_ind = linear_sum_assignment(-ious)

        for r, c in zip(row_ind, col_ind):
            if ious[r, c] >= iou_threshold:
                gid = gt_ids[r]
                pid = pred_ids[c]
                color = colors[color_idx % len(colors)]

                gt_rgb[gt_mask == gid] = color
                pred_rgb[pred_mask == pid] = color

                matched_gt.add(gid)
                matched_pred.add(pid)
                color_idx += 1

    for gid in gt_ids:
        if gid not in matched_gt:
            gt_rgb[gt_mask == gid] = colors[color_idx % len(colors)]
            color_idx += 1

    for pid in pred_ids:
        if pid not in matched_pred:
            pred_rgb[pred_mask == pid] = colors[color_idx % len(colors)]
            color_idx += 1

    return gt_rgb, pred_rgb


def plot_and_save_images(
    imgs,
    gt_masks,
    pred_masks,
    output_img_path,
    iou_threshold=0.1,
    background_id=0,
    dpi=150,
):
    n = len(imgs)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)

    col_titles = ["Sample Image", "Ground truth", "Predicted Mask"]

    for row in range(n):
        img = imgs[row]
        gt_mask = gt_masks[row]
        pred_mask = pred_masks[row]

        gt_vis, pred_vis = _matched_color_maps(
            gt_mask,
            pred_mask,
            iou_threshold=iou_threshold,
            background_id=background_id,
        )

        panels = [img, gt_vis, pred_vis]

        for col in range(3):
            ax = axes[row, col]
            ax.imshow(panels[col])
            ax.axis("off")

            if row == 0:
                ax.set_title(col_titles[col], fontsize=12)

    plt.tight_layout()
    plt.savefig(output_img_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

@torch.no_grad()
def evaluate(model, val_loader, device, approach, output_dir):
    sample_img_save_indices = random.sample(range(len(val_loader)), k=3)
    sample_img_path = os.path.join(output_dir, "sample_img.png")
    sample_imgs = []
    sample_masks = []
    sample_preds = []

    dice_scores = []
    aji_scores = []
    pq_scores = []

    model.eval()

    idx = 0

    for img, gt_label, image_np in tqdm(val_loader, "Validating"):
        pixel_values = img.to(device)
        gt_label_np = gt_label.squeeze(0).cpu().numpy().astype(np.int32)

        image_np_single = image_np.squeeze(0).cpu().numpy()

        if approach == "auto":
            points = get_grid_points(image_size=512, points_per_side=32)
        elif approach == "point":
            points = sample_one_point_per_instance(gt_label_np)
        else:
            raise ValueError(f"Approach: {approach} not currently supported for prompting")

        pred_label = predict_instances_with_points(
            model=model,
            pixel_values=pixel_values,
            points=points,
            orig_size=gt_label_np.shape[-1],
            target_size=1024,
            out_size=gt_label_np.shape[-1],  # 256
            device=device,
            score_threshold=0.50,
            nms_iou=0.3
        )

        dice_scores.append(binary_dice(pred_label, gt_label_np))
        aji_scores.append(aji_score(gt_label_np, pred_label))
        pq_scores.append(pq_score(gt_label_np, pred_label))

        if idx in sample_img_save_indices:
            resized_img = cv2.resize(image_np_single, (gt_label_np.shape[0], gt_label_np.shape[1]))
            sample_imgs.append(resized_img)
            sample_masks.append(gt_label_np)
            sample_preds.append(pred_label)

    plot_and_save_images(imgs=sample_imgs, gt_masks=sample_masks, pred_masks=sample_preds, output_img_path=sample_img_path)

    return {
        "dice": float(np.mean(dice_scores)),
        "aji": float(np.mean(aji_scores)),
        "pq": float(np.mean(pq_scores)),
    }

def main():
    args = get_args()

    set_seed(args.seed)

    img_ids = get_all_image_ids(root=args.root)

    folds_img_ids = create_folds(img_ids=img_ids, num_folds=5)

    device = get_device()

    mask_transform = get_mask_transforms()

    model, processor = get_model(sam_path=args.sam_path)
    model = PeftModel.from_pretrained(model, args.lora_path)
    
    model = model.to(device=device)

    _, val_img_ids = get_train_val_img_ids(fold_img_ids=folds_img_ids, val_idx=args.fold)
    val_ds = NucleiDataset(root=args.root, return_instances_separately=False, image_ids=val_img_ids, img_transform=processor, label_transform=mask_transform)

    val_loader = DataLoader(dataset=val_ds, shuffle=False, batch_size=1)

    metrics = evaluate(model=model, val_loader=val_loader, device=device, approach=args.approach, output_dir=args.output_dir)
    print(metrics)

if __name__ == "__main__":
    main()