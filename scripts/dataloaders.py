import argparse
import logging

import torchvision

import nibabel as nib

import fnmatch
import os
import random
import shutil
import string
import time
from abc import abstractmethod
from collections import defaultdict
from time import sleep
import torch
import cv2

import numpy as np
from torch.utils.data import DataLoader, Dataset
import transforms

# Deffine transforms for BRAIN and ABDOMEN scans
BRAIN_DEFAULT_TRANSFORM = torchvision.transforms.Compose(
        [   
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128,128), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.ToTensor(),
            transforms.Scale(a=0, b=255, min_val=0, max_val=1),  # Scale to [0, 1]
            transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 256]
            transforms.Scale(a=0, b=1, min_val=0, max_val=256),  # Scale to [0, 1]
        ]
    )

ABDOM_DEFAULT_TRANSFORM = torchvision.transforms.Compose(
        [   
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((256,256), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.ToTensor(),
            transforms.Scale(a=0, b=255, min_val=0, max_val=1),  # Scale to [0, 1]
            transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 256]
            transforms.Scale(a=0, b=1, min_val=0, max_val=256),  # Scale to [0, 1]
        ]
    )

class MOOD2dDataSet(Dataset):
    def __init__(
        self,
        root = "",
        transform=None,
    ):
        super().__init__()

        """Dataset which loads 2D slices from a dir of NIFTI files.

        Args:
            base_dir ([str]): [Directory in which the nifti files are.]
            transforms ([type], optional): [Transformation to do after loading the dataset -> pytorch data transforms]. Defaults to None
        """
        
        self.root = root
        self.transform = transform

        self.items = self.load_dataset(self.root)
        self.data_len = len(self.items)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data_smpl, fn = self.get_data_by_idx(idx)

        data_smpl = data_smpl.transpose(1,2,0)

        data_smpl = [self.transform(data_smpl[i]) for i in range(data_smpl.shape[0])]
        data_smpl = [torch.cat([data_smpl[i]]*3, dim=0) for i in range(len(data_smpl))]
        data_smpl = torch.stack(data_smpl, dim=0)
                 
        return data_smpl, fn

    def get_data_by_idx(self, idx):
        """Returns a data sample for a given index i.e., file path -> 3-D Numpy array DxHxW

        Args:
            idx ([int]): [Index of the data sample]

        Returns:
            [np.ndarray]: [3-D Numpy array DxHxW]
        """

        file = self.items[idx]
        nifti = nib.load(file)
        np_data = nifti.get_fdata()
        np_data = np_data.astype(np.float16)
    

        # slice to get 1 slice  
        return np_data.astype(np.float32), file
   

    def load_dataset(self, base_dir):
        """Indexes all files in the given directory and returns a list of 2-D slices (file_index, npy_file, slice_index_for_np_file)
          (so they can be loaded with get_data_by_idx)

        Args:
            base_dir ([str]): [Directory in which the npy files are.]
        Returns:
            [list]: [List of file paths which should be used in the dataset]
        """
        files = []
        all_files = os.listdir(base_dir)
        
        for i, filename in enumerate(sorted(all_files)):
            if not filename.endswith("nii.gz"):
                continue

            n_file = os.path.join(base_dir, filename)
            files.append(n_file)
        
        return files
    



    