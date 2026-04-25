from NucleiDataset import NucleiDataset
import argparse
from torch.utils.data import DataLoader
from utils import box_from_mask
from torchvision import transforms
import torchvision.transforms.functional as TF
from transformers import SamModel, SamProcessor
import torch
import numpy as np

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    model_name = "facebook/sam-vit-base"

    processor = SamProcessor.from_pretrained(model_name)

    ds = NucleiDataset(root=args.root, return_instances_separately=True, img_transform=processor, label_transform=get_mask_transforms())
    loader = DataLoader(dataset=ds, shuffle=True, batch_size=32)

    for img, label, img_np in loader:
        input_boxes = box_from_mask(label)
        print(img.shape, label.shape, img_np.shape, input_boxes.shape)


    print("All set")


if __name__ == "__main__":
    main()
    
