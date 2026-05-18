# sartorius_utils.py
import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def rle_decode(mask_rle: str, shape: tuple) -> np.ndarray:
    s       = list(map(int, mask_rle.split()))
    starts  = np.array(s[0::2]) - 1
    lengths = np.array(s[1::2])
    ends    = starts + lengths
    img     = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    return img.reshape(shape)


def normalize_simple(img: np.ndarray) -> np.ndarray:
    return (img / 255.0).astype(np.float32)


def normalize_imagenet(img: np.ndarray,
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)) -> np.ndarray:
    img  = img.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std  = np.array(std,  dtype=np.float32)
    return (img - mean) / std


def resize_image(img: np.ndarray, size=(512, 512)) -> np.ndarray:
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, size=(512, 512)) -> np.ndarray:
    return cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


def load_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def random_horizontal_flip(img, masks, p=0.5):
    if random.random() < p:
        img   = np.fliplr(img).copy()
        masks = np.flip(masks, axis=2).copy()
    return img, masks


def random_vertical_flip(img, masks, p=0.5):
    if random.random() < p:
        img   = np.flipud(img).copy()
        masks = np.flip(masks, axis=1).copy()
    return img, masks


def random_rotation_90(img, masks):
    k     = random.randint(0, 3)
    img   = np.rot90(img,   k=k, axes=(0, 1)).copy()
    masks = np.rot90(masks, k=k, axes=(1, 2)).copy()
    return img, masks


def random_zoom_crop(img, masks, scale_range=(0.8, 1.0), target_size=(512, 512)):
    H, W      = img.shape[:2]
    scale     = random.uniform(*scale_range)
    new_H, new_W = int(H * scale), int(W * scale)
    top       = random.randint(0, H - new_H)
    left      = random.randint(0, W - new_W)
    img_crop  = img[top:top+new_H, left:left+new_W]
    masks_crop = masks[:, top:top+new_H, left:left+new_W]
    img_out   = cv2.resize(img_crop, (target_size[1], target_size[0]),
                            interpolation=cv2.INTER_LINEAR)
    masks_out = np.stack([
        cv2.resize(m, (target_size[1], target_size[0]),
                   interpolation=cv2.INTER_NEAREST)
        for m in masks_crop
    ], axis=0)
    return img_out, masks_out


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1))


def preprocess(img_path, rle_list, target_size=(512, 512),
               is_train=True, use_elastic=False,
               normalization='imagenet') -> dict:
    img = load_image_rgb(img_path)

    masks_list = [rle_decode(rle, (img.shape[0], img.shape[1]))
                  for rle in rle_list if not pd.isna(rle) and rle != '']
    if len(masks_list) == 0:
        masks_np = np.zeros((1, img.shape[0], img.shape[1]), dtype=np.uint8)
    else:
        masks_np = np.stack(masks_list, axis=0)

    img      = resize_image(img, size=target_size)
    masks_np = np.stack([resize_mask(m, size=target_size) for m in masks_np])

    if is_train:
        img, masks_np = random_horizontal_flip(img, masks_np)
        img, masks_np = random_vertical_flip(img, masks_np)
        img, masks_np = random_rotation_90(img, masks_np)
        img, masks_np = random_zoom_crop(img, masks_np, target_size=target_size)
        if use_elastic:
            from scipy.ndimage import gaussian_filter
            H, W  = img.shape[:2]
            alpha, sigma = 34.0, 4.0
            dx    = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma) * alpha
            dy    = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma) * alpha
            x, y  = np.meshgrid(np.arange(W), np.arange(H))
            map_x = np.clip(x + dx, 0, W - 1).astype(np.float32)
            map_y = np.clip(y + dy, 0, H - 1).astype(np.float32)
            img      = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
            masks_np = np.stack([
                cv2.remap(m, map_x, map_y, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REFLECT_101)
                for m in masks_np
            ], axis=0)

    img_float = normalize_imagenet(img) if normalization == 'imagenet' \
                else normalize_simple(img)

    return {
        'image' : image_to_tensor(img_float),
        'masks' : torch.from_numpy(masks_np),
        'meta'  : {'path': img_path, 'n_masks': len(masks_list),
                   'size': target_size}
    }


class SartoriusDataset(Dataset):
    def __init__(self, df, is_train=True, target_size=(512, 512),
                 normalization='imagenet', use_elastic=False):
        if 'ruta_imagen' not in df.columns:
            raise ValueError("El DataFrame debe tener la columna 'ruta_imagen'. "
                             "Pasa df_img directamente o haz el merge antes.")
        self.df            = df.reset_index(drop=True)
        self.is_train      = is_train
        self.target_size   = target_size
        self.normalization = normalization
        self.use_elastic   = use_elastic

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        result = preprocess(
            img_path      = row['ruta_imagen'],
            rle_list      = row['rles'],
            target_size   = self.target_size,
            is_train      = self.is_train,
            use_elastic   = self.use_elastic,
            normalization = self.normalization
        )
        result['meta']['id']        = row['id_imagen']
        result['meta']['cell_type'] = row['cell_type']
        return result