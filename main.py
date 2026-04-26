import argparse
import os
import torch
import json
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
from utils import pq_score, binary_dice, aji_score, mask_iou, get_grid_points, generate_proposal_boxes_from_image
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sam_path", default=None, type=str)

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

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,                                   
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(model, peft_config)
    return model, processor

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    

@torch.no_grad()
def predict_instances_with_boxes(
    model,
    pixel_values: torch.Tensor,      
    boxes_xyxy: torch.Tensor,
    device: torch.device,
    orig_size: int = 512,
    mask_size: int = 256,
    mask_threshold: float = 0.0,
):
    model.eval()

    if boxes_xyxy.numel() == 0:
        return np.zeros((mask_size, mask_size), dtype=np.int32)

    pixel_values = pixel_values.to(device)

    scale = pixel_values.shape[-1] / orig_size
    input_boxes = boxes_xyxy.to(device) * scale
    input_boxes = input_boxes.unsqueeze(0)  # [1,N,4]

    outputs = model(
        pixel_values=pixel_values,
        input_boxes=input_boxes,
        multimask_output=False,
    )

    masks = outputs.pred_masks[0, :, 0]
    masks = masks > mask_threshold

    pred_label = np.zeros((mask_size, mask_size), dtype=np.int32)

    for i, m in enumerate(masks):
        m_np = m.detach().cpu().numpy().astype(bool)

        m_np = np.logical_and(m_np, pred_label == 0)

        if m_np.sum() == 0:
            continue

        pred_label[m_np] = i + 1

    return pred_label

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
    
@torch.no_grad()
def evaluate(model, val_loader, device):
    dice_scores = []
    aji_scores = []
    pq_scores = []

    model.eval()

    for img, gt_label, image_np in tqdm(val_loader, "Validating"):
        pixel_values = img.to(device)
        gt_label_np = gt_label.squeeze(0).cpu().numpy().astype(np.int32)

        image_np_single = image_np.squeeze(0).cpu().numpy()
        points = get_grid_points(image_size=512, points_per_side=64)

        # boxes = generate_proposal_boxes_from_image(image_np_single)

        pred_label = predict_instances_with_points(
            model=model,
            pixel_values=pixel_values,
            points=points,
            orig_size=512,
            target_size=1024,
            out_size=gt_label_np.shape[-1],  # 56
            device=device,
            score_threshold=0.50,
            nms_iou=0.3
        )

        dice_scores.append(binary_dice(pred_label, gt_label_np))
        aji_scores.append(aji_score(gt_label_np, pred_label))
        pq_scores.append(pq_score(gt_label_np, pred_label))

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

    for val_idx in range(len(folds_img_ids)):
        model, processor = get_model(sam_path=args.sam_path)
        model = model.to(device=device)

        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=1e-4,
        )
        seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        bce_loss = torch.nn.BCEWithLogitsLoss()

        train_img_ids, val_img_ids = get_train_val_img_ids(fold_img_ids=folds_img_ids, val_idx=val_idx)

        train_ds = NucleiDataset(root=args.root, return_instances_separately=True, image_ids=train_img_ids, img_transform=processor, label_transform=mask_transform)
        indices = random.sample(population=range(len(train_ds)), k=15000)
        train_ds = Subset(train_ds, indices=indices)

        val_ds = NucleiDataset(root=args.root, return_instances_separately=False, image_ids=val_img_ids, img_transform=processor, label_transform=mask_transform)

        train_loader = DataLoader(dataset=train_ds, shuffle=True, batch_size=args.batch_size)
        val_loader = DataLoader(dataset=val_ds, shuffle=False, batch_size=1)

        best_score = 0.0

        fold_dir = os.path.join(args.output_dir, str(val_idx))
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        val_metrics = []
        train_losses = []

        for epoch in tqdm(range(args.epochs), desc=f"Training fold configuration: {val_idx + 1}"):
            avg_loss = num_samples = 0.0
            model.train()
            for img, gt_mask, _ in tqdm(train_loader, desc=f"Epoch: {epoch + 1}"):
                pixel_values = img.to(device)
                gt_mask = gt_mask.to(device=device)

                input_boxes = box_from_mask(gt_mask)
                scale = pixel_values.shape[-1] / gt_mask.shape[-1]
                input_boxes = (input_boxes * scale).to(device)

                optim.zero_grad(set_to_none=True)

                outputs = model(
                    pixel_values=pixel_values,
                    input_boxes=input_boxes,
                    multimask_output=False
                )

                predicted_masks = outputs.pred_masks.squeeze(1).squeeze(1)
                gt_mask = gt_mask.float()

                loss_dice = seg_loss(predicted_masks.unsqueeze(1), gt_mask.unsqueeze(1))
                loss_bce = bce_loss(predicted_masks, gt_mask)

                loss = loss_dice + loss_bce

                loss.backward()
                optim.step()

                avg_loss += loss.detach().cpu().item() * pixel_values.size(0)
                num_samples += pixel_values.size(0)

            avg_loss /= num_samples
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

            metrics = evaluate(model=model, val_loader=val_loader, device=device)

            print(f"Epoch {epoch + 1} | Val metrics: {metrics}")

            train_losses.append(avg_loss)
            val_metrics.append(metrics)

            if metrics["pq"] > best_score:
                best_score = metrics["pq"]
                model.save_pretrained(os.path.join(fold_dir, "best_lora"))
        

        with open(os.path.join(fold_dir, "logs.json"), "w") as f:
            json.dump({
                "train_losses" : train_losses,
                "val_metrics": val_metrics
            }, f)

if __name__ == "__main__":
    main()