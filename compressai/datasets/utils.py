# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import glob
import os
import datetime
import random

class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')
        
        self.samples = glob.glob(os.path.join(splitdir, '*.jpg'))
        self.samples += glob.glob(os.path.join(splitdir, '*.png'))

        # self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

        self.split = split

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        try:
            img = Image.open(self.samples[index]).convert("RGB")
        except:
            print("loading error:%s"%self.samples[index])
            while True:
                index = index = random.randint(0, len(self) - 1)
                try:
                    img = Image.open(self.samples[index]).convert("RGB")
                    break
                except:
                    print("loading error:%s"%self.samples[index])
                
        if self.split == "test":
            imsize = img.size
            if imsize[0]>imsize[1]:
                img = img.transpose(Image.ROTATE_90)

        if self.transform:
            return self.transform(img)
        
        return img

    def __len__(self):
        return len(self.samples)

