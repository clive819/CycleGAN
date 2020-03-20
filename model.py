from config import DenseNetConfig as dnc
import torch
import torch.nn as nn


class _transitionLayer(nn.Module):
    def __init__(self, inChannels, downSample=True):
        super(_transitionLayer, self).__init__()

        self.outChannels = int(inChannels * dnc.compressionRatio)

        layers = [
            nn.GroupNorm(dnc.numGroups, inChannels),
            nn.ReLU(),
        ]

        if downSample:
            layers.append(nn.Conv2d(inChannels, self.outChannels, 3, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(inChannels, 4 * self.outChannels, 1))
            layers.append(nn.PixelShuffle(2))

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class _convBlock(nn.Module):
    def __init__(self, inChannels):
        super(_convBlock, self).__init__()

        self.outChannels = dnc.growthRate
        self.module = nn.Sequential(
            nn.GroupNorm(dnc.numGroups, inChannels),
            nn.ReLU(),
            nn.Conv2d(inChannels, 4 * dnc.growthRate, 1),
            nn.GroupNorm(dnc.numGroups, 4 * dnc.growthRate),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(4 * dnc.growthRate, dnc.growthRate, 3)
        )

    def forward(self, x):
        return self.module(x)


class _denseBlock(nn.Module):
    def __init__(self, inChannels, numBlocks):
        super(_denseBlock, self).__init__()

        self.outChannels = inChannels

        self.layers = nn.ModuleList()
        for _ in range(numBlocks):
            self.layers.append(_convBlock(self.outChannels))
            self.outChannels += dnc.growthRate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, 1)))

        return torch.cat(features, 1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        outChannels = 64

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, outChannels, 7),
            nn.GroupNorm(dnc.numGroups, outChannels),
            nn.ReLU(),
        ]

        for b in dnc.generatorBlocks:
            db = _denseBlock(outChannels, b)
            outChannels = db.outChannels
            t = _transitionLayer(outChannels, downSample=True)
            outChannels = t.outChannels
            layers.append(db)
            layers.append(t)

        for _ in range(len(dnc.generatorBlocks)):
            t = _transitionLayer(outChannels, downSample=False)
            outChannels = t.outChannels
            layers.append(t)

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(outChannels, 3, 7),
            nn.Sigmoid()
        ]

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        outChannels = 64

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, outChannels, 7),
            nn.GroupNorm(dnc.numGroups, outChannels),
            nn.ReLU(),
        ]

        for numBlock in dnc.discriminatorBlocks:
            db = _denseBlock(outChannels, numBlock)
            outChannels = db.outChannels
            t = _transitionLayer(outChannels, downSample=True)
            outChannels = t.outChannels
            layers.append(db)
            layers.append(t)

        layers += [
            nn.Conv2d(outChannels, 1, 1)
        ]

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)
