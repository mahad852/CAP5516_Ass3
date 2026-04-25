from NucleiDataset import NucleiDataset
import argparse
from torch.utils.data import DataLoader, Subset
from utils import box_from_mask
from torchvision import transforms
import torchvision.transforms.functional as TF
from transformers import SamModel, SamProcessor
import torch
import numpy as np
from tqdm import tqdm
import random
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    set_seed(42)
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    model_name = "facebook/sam-vit-base"

    processor = SamProcessor.from_pretrained(model_name)

    ds = NucleiDataset(root=args.root, return_instances_separately=True, img_transform=processor, label_transform=get_mask_transforms())
    # indices = range(22500, 22600)
    # ds = Subset(ds, indices=indices)

    loader = DataLoader(dataset=ds, shuffle=True, batch_size=32)

    for idx, (img, label, img_np) in tqdm(enumerate(loader), desc="Running test"):
        print(img.shape, label.shape, img_np.shape)
        input_boxes = box_from_mask(label)
        if input_boxes.size(0) != label.size(0) != img.size(0) != img_np.size(0):
            print(img.shape, label.shape, img_np.shape, input_boxes.shape)
            raise ValueError(f"{idx}, all tensors must have the same num batches")

        if img.size(2) != img.size(3) or img.size(2) != 1024 or img.size(1) != 3:
            print(img.shape, label.shape, img_np.shape, input_boxes.shape)
            raise ValueError(f"{idx}, img tensor must be of size [B, 3, 1024, 1024]") 
        
        if label.size(1) != label.size(2) or label.size(1) != 256:
            print(img.shape, label.shape, img_np.shape, input_boxes.shape)
            raise ValueError(f"{idx}, label tensor must be of size [B, 256, 256]") 
        
        if input_boxes.size(1) != 1 or input_boxes.size(2) != 4:
            print(img.shape, label.shape, img_np.shape, input_boxes.shape)
            raise ValueError(f"{idx}, input_boxes tensor must be of size [B, 1, 4]") 

    print("All set")


if __name__ == "__main__":
    main()
    
