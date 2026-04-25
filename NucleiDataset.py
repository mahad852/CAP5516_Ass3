from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class NucleiDataset(Dataset):
    def __init__(self, root: str, img_transform = None, label_transform = None, return_instances_separately = True, image_ids = None):
        self.root = root
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.image_ids = image_ids
        self.return_instances_separately = return_instances_separately

        self.index = self._build_index()

    def get_mask_instance_ids(self, mask_path: str):
        mask = Image.open(mask_path)
        instances_ids = np.unique(mask).tolist()
        return [instance_id for instance_id in instances_ids if instance_id > 0]

    def _build_index(self):
        index = []
        for organ_dir in os.listdir(self.root):
            if organ_dir.startswith(".") or organ_dir.startswith("_"):
                continue
            label_dir = os.path.join(self.root, organ_dir, "label masks modify")
            image_dir = os.path.join(self.root, organ_dir, "tissue images")

            for image_fname in os.listdir(image_dir):
                img_id = image_fname.split(".")[0]

                if self.image_ids and img_id not in self.image_ids:
                    continue

                label_fname = f"{img_id}.tif"
                image_path = os.path.join(image_dir, image_fname)
                label_path = os.path.join(label_dir, label_fname)

                if self.return_instances_separately:
                    instance_ids = self.get_mask_instance_ids(mask_path=label_path)
                    for instance_id in instance_ids:
                        index.append((image_path, label_path, instance_id))

                else:
                    index.append((image_path, label_path, None))

        return index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        image_path, label_path, instance_id = self.index[index]

        img = Image.open(image_path)

        if self.img_transform:
            transformed_img = self.img_transform(img)

        label = Image.open(label_path)
        if self.label_transform:
            label = self.label_transform(label)

        if instance_id:
            label = label == instance_id

        return transformed_img, label, np.array(img)


