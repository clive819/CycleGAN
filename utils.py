from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image
import os
import torch
import numpy as np


class CGANDataset(Dataset):
    def __init__(self, root, _transforms, prefix='train'):

        self.a = glob(os.path.join(root, '{}A/'.format(prefix), '*.jpg'))
        self.b = glob(os.path.join(root, '{}B/'.format(prefix), '*.jpg'))

        self.transforms = _transforms

    def __len__(self):
        return max(len(self.a), len(self.b))

    def __getitem__(self, idx):
        a = Image.open(self.a[idx % len(self.a)]).convert('RGB')
        b = Image.open(self.b[idx % len(self.b)]).convert('RGB')

        return self.transforms(a), self.transforms(b)


class RelayBuffer(object):
    def __init__(self, poolSize=50):
        self.poolSize = poolSize
        self.pool = []

    def query(self, images):
        ans = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if len(self.pool) < self.poolSize:
                ans.append(image)
                self.pool.append(image)
            else:
                if np.random.uniform(0, 1) > .5:
                    idx = np.random.randint(0, self.poolSize - 1)
                    ans.append(self.pool[idx].clone())
                    self.pool[idx] = image
                else:
                    ans.append(image)
        return torch.cat(ans, 0)
