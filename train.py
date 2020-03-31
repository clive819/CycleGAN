from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from model import Generator, Discriminator
from utils import RelayBuffer, CGANDataset
from torch.optim import Adam
from itertools import chain
from config import device
import torch
import numpy as np


# MARK: - hyperparamters
cropSize = 256

dataTransforms = transforms.Compose([
    transforms.RandomCrop(cropSize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


# MARK: - load data
dataset = CGANDataset('/Users/clive/Downloads/horse2zebra', dataTransforms, prefix='train')
dataLoader = DataLoader(dataset, batch_size=1, shuffle=True)


# MARK: - train
writer = SummaryWriter('./logs')
prevBestLoss = np.inf
batches = len(dataLoader)

generatorA2B, generatorB2A = Generator().to(device), Generator().to(device)
discriminatorA, discriminatorB = Discriminator().to(device), Discriminator().to(device)

bufferA = RelayBuffer()
bufferB = RelayBuffer()

gOptimizer = Adam(chain(generatorA2B.parameters(), generatorB2A.parameters()), lr=1e-4)
dOptimizer = Adam(chain(discriminatorA.parameters(), discriminatorB.parameters()), lr=1e-4)

l1 = torch.nn.L1Loss()
mse = torch.nn.MSELoss()

for epoch in range(1000):
    gLoss = []
    for batch, (A, B) in enumerate(dataLoader):
        A, B = A.to(device), B.to(device)

        # MARK: - forward
        fakeB = generatorA2B(A)
        reconstructedA = generatorB2A(fakeB)

        fakeA = generatorB2A(B)
        reconstructedB = generatorA2B(fakeA)

        # MARK: - update generator
        discriminatorA.requires_grad_(False)
        discriminatorB.requires_grad_(False)
        gOptimizer.zero_grad()

        # identity loss (Sec 5.2)
        idA = generatorA2B(B)
        lossIDA = l1(idA, B) * 5.

        idB = generatorB2A(A)
        lossIDB = l1(idB, A) * 5.

        # GAN loss
        da = discriminatorB(fakeB)
        lossGA2B = mse(da, torch.ones_like(da))

        db = discriminatorA(fakeA)
        lossGB2A = mse(db, torch.ones_like(db))

        lossCycleA = l1(reconstructedA, A) * 10.
        lossCycleB = l1(reconstructedB, B) * 10.

        lossG = lossIDA + lossIDB + lossGA2B + lossGB2A + lossCycleA + lossCycleB
        lossG.backward()

        gLoss.append(lossG.cpu().item())

        gOptimizer.step()

        # MARK: - update discriminator
        discriminatorA.requires_grad_(True)
        discriminatorB.requires_grad_(True)
        dOptimizer.zero_grad()

        # update discriminator A
        fakeA = bufferA.query(fakeA)

        predReal = discriminatorA(A)
        lossReal = mse(predReal, torch.ones_like(predReal))

        predFake = discriminatorA(fakeA.detach())
        lossFake = mse(predFake, torch.zeros_like(predFake))

        lossDA = (lossReal + lossFake) / 2.
        lossDA.backward()

        # update discriminator B
        fakeB = bufferB.query(fakeB)

        predReal = discriminatorB(B)
        lossReal = mse(predReal, torch.ones_like(predReal))

        predFake = discriminatorB(fakeB.detach())
        lossFake = mse(predFake, torch.zeros_like(predFake))

        lossDB = (lossReal + lossFake) / 2.
        lossDB.backward()

        dOptimizer.step()

        print('\rEpoch {}, {} / {}, lossG: {:.8f}'.format(epoch, batch, batches, gLoss[-1]), end='')

    avgLoss = np.mean(gLoss)
    print('Epoch {}, Generator loss: {:.8f}'.format(epoch, avgLoss))
    writer.add_scalar('Loss/Train', avgLoss, epoch)

    if avgLoss < prevBestLoss:
        print('[+] Loss improved from {:.8f} to {:.8f}, saving model...'.format(prevBestLoss, avgLoss))
        prevBestLoss = avgLoss
        torch.save(generatorA2B.state_dict(), 'modelA2B.pt')
        torch.save(generatorB2A.state_dict(), 'modelB2A.pt')
        torch.save(discriminatorA.state_dict(), 'discriminatorA.pt')
        torch.save(discriminatorB.state_dict(), 'discriminatorB.pt')
        writer.add_scalar('Loss/Model', avgLoss, epoch)

    writer.flush()
writer.close()




