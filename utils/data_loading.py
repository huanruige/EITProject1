import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        img_ndarray = np.asarray(pil_img)
        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))
        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return pd.read_csv(filename, header=None)

    def __getitem__(self, idx):
        name = self.ids[idx]
        # mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        # img_file = list(self.images_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        # 替换 _ddl
        mask_stem = name.replace('_bv', '_ddl')  # '0780_ddl'
        mask_file = list(self.masks_dir.glob(mask_stem + '.*'))  # '0780_ddl.*' → 匹配 '0780_ddl.csv'

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='')


class RealGNDataset(BasicDataset):
    """
    用于实测+GN 微调的数据集：
    - images_dir: 存 BV_norm，文件名形如 '0780_bv.csv'
    - masks_dir:  存 GN 的 DDL_norm，文件名形如 '0780_ddl.csv'
    命名规则与模拟数据完全一致，只是目录不同。
    """

    def __init__(self, images_dir, masks_dir, scale=1.0):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='')

    def __getitem__(self, idx):
        name = self.ids[idx]

        # image = BV_norm，和模拟时一样，直接用 name 找
        img_file = list(self.images_dir.glob(name + '.*'))

        # mask = GN 重建后的 DDL_norm
        # 文件名规则仍然是: 0780_bv.csv -> 0780_ddl.csv
        mask_stem = name.replace('_bv', '_ddl')
        mask_file = list(self.masks_dir.glob(mask_stem + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        mask = self.load(mask_file[0])  # DataFrame / ndarray
        img  = self.load(img_file[0])

        img  = self.preprocess(img,  self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            # 输入 BV：和模拟时一样，float
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            # 输出 DDL：这里很重要，作为“回归值”，用 float，不再是 long
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }

