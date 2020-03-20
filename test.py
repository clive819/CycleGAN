from model import Generator, _transitionLayer, _convBlock, _denseBlock
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from utils import CGANDataset
from config import device

import torch


dataset = CGANDataset('/Users/clive/Downloads/horse2zebra', transforms.ToTensor(), prefix='test')
dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)


modelA2B = torch.load('./modelA2B.pt', map_location=device)


modelA2B.eval()
with torch.no_grad():
    for x, y in dataLoader:
        out = modelA2B(x.to(device))
        img = transforms.ToPILImage()(torch.squeeze(out.cpu()))
        img.show()
        transforms.ToPILImage()(torch.squeeze(x)).show()
        transforms.ToPILImage()(torch.squeeze(y)).show()
        a=1


