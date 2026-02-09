import os
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import (Compose, LoadImaged, ToTensord,
                              RandFlipd, RandRotated, RandZoomd,
                              RandGaussianNoised, RandGaussianSmoothd,
                              CastToTyped, EnsureTyped,ResizeD,NormalizeIntensityd)
from skimage.transform import resize
from scipy.ndimage import label
import math
import torch
import re
from torch.utils.data import Dataset as TorchDataset #
import copy
import monai

class Dataset2D(Dataset):
    def __init__(self, DATASET, target_shape=(224, 224), augmentations=False):
        self.target_shape = target_shape
        self.augmentations_flag = augmentations
        self.items = []

        for dataset in DATASET:
            mask_folder = dataset + "_Masks"

            for label_name, label_value in {"0": 0, "1": 1}.items():
                subfolder_path = os.path.join(dataset, label_name)
                if not os.path.exists(subfolder_path):
                    continue

                for f in os.listdir(subfolder_path):
                    if f.endswith(".npy"):
                        match = re.search(r"_(\d+)", f)
                        file_id = int(match.group(1)) if match else -1
                        self.items.append({
                            "image": os.path.join(subfolder_path, f),
                            "label": label_value,
                            "id": file_id,
                            "dataset": os.path.basename(dataset)
                        })

        if len(self.items) == 0:
            raise RuntimeError("Nenhum .npy encontrado.")



class RuntimeAugDataset(TorchDataset):
    def __init__(self, cached_ds, indices, transform=None):
        self.cached_ds = cached_ds
        self.indices = indices
        self.transform = transform
        self.seed = 1
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data = copy.deepcopy(self.cached_ds[real_idx]) 
        self.seed += 1
        monai.utils.set_determinism(seed=self.seed)

        if self.transform:
            data = self.transform(data)

        return data
       
