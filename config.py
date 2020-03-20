import torch
import numpy as np


torch.manual_seed(1588390)
np.random.seed(1588390)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DenseNetConfig(object):
    # number of groups for group normalization
    numGroups = 8

    compressionRatio = .5

    growthRate = 32

    generatorBlocks = [6, 12]

    discriminatorBlocks = [4, 4, 4, 4]
