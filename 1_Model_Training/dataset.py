# 1_Model_Training/dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset

class OxfordPetsDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset loader using trainval/test txt files.
    """
    def __init__(self, root, split="trainval", transform=None):
        """
        root: path to dataset root containing 'images/' and 'annotations/'
        split: 'trainval' or 'test'
        """
        self.root = root
        self.transform = transform

        annot_file = os.path.join(root, "annotations", f"{split}.txt")
        self.samples = []

        with open(annot_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                img_id = parts[0]           # e.g. 'Abyssinian_1'
                class_id = int(parts[1])    # 1..37

                img_path = os.path.join(root, "images", img_id + ".jpg")
                label = class_id - 1        # convert to 0..36
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
