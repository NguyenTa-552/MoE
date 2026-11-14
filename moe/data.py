"""Explicit annotation modes; no hidden broadcast of image labels."""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class ManifestDataset(Dataset):
    def __init__(self, rows, root, baseline, label_ids, supervision, ignore_index=-1):
        if supervision not in ('typed_mask', 'binary_mask_with_image_type'):
            raise ValueError('This dense implementation requires verified spatial masks; image-list-only mode is not supported')
        self.rows, self.root, self.baseline = rows, Path(root), baseline
        self.label_ids, self.supervision, self.ignore_index = label_ids, supervision, ignore_index
        for row in rows:
            if not (self.root / row['img_path']).is_file():
                raise FileNotFoundError(self.root / row['img_path'])
            if row['anomaly'] and not (self.root / row['mask_path']).is_file():
                raise FileNotFoundError(self.root / row['mask_path'])
            if supervision == 'binary_mask_with_image_type' and row['anomaly'] and row['defect'] not in label_ids:
                raise ValueError(f'Unmapped source defect: {row["defect"]}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.root / row['img_path']).convert('RGB')
        pixels = self.baseline.preprocess(image)
        if not row['anomaly']:
            mask = torch.zeros(pixels.shape[-2:], dtype=torch.long)
            baseline_mask = mask.clone()
        elif self.supervision == 'typed_mask':
            # ID-preserving resize is explicit; ignored IDs remain ignored.
            raw = Image.open(self.root / row['mask_path'])
            size = list(pixels.shape[-2:])
            raw = TF.center_crop(TF.resize(raw, size, InterpolationMode.NEAREST), size)
            mask = torch.from_numpy(np.array(raw, dtype=np.int64).copy())
            baseline_mask = mask.clone()
        else:
            raw = np.array(Image.open(self.root / row['mask_path']).convert('L')) > 0
            raw = Image.fromarray(raw.astype(np.uint8) * 255, mode='L')
            resized = self.baseline.mask_transform(raw).squeeze(0)
            # Preserve separate historical training and evaluation thresholds.
            baseline_mask = (resized.long() > 0).long() * self.label_ids[row['defect']]
            mask = (resized > 0.5).long() * self.label_ids[row['defect']]
            if raw.getbbox() is not None and not mask.any():
                raise ValueError(f'Defect vanished during mask preprocessing: {row["image_id"]}; inspect tiny defects before training')
        return dict(image=pixels, mask=mask, training_mask=baseline_mask,
                    anomaly=torch.tensor(row['anomaly']), index=index)
