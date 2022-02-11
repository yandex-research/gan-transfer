from torch import nn
from gans.models.playground.data import SAMPLES_DIM


class Generator(nn.Module):
    def __init__(self, width=64):
        super(Generator, self).__init__()

        self.dim_z = 64
        self.module = nn.Sequential(
            nn.Linear(self.dim_z, width), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Linear(width, 2 * width), nn.BatchNorm1d(2 * width), nn.ReLU(),
            nn.Linear(2 * width, 2 * width), nn.BatchNorm1d(2 * width), nn.ReLU(),
            nn.Linear(2 * width, 2 * width), nn.BatchNorm1d(2 * width), nn.ReLU(),
            nn.Linear(2 * width, width), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Linear(width, SAMPLES_DIM),
        )

    def forward(self, z):
        return self.module(z)


class Discriminator(nn.Module):
    def __init__(self, width=64, return_probs=True):
        super(Discriminator, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(SAMPLES_DIM, width), nn.ReLU(),
            nn.Linear(width, 2 * width), nn.ReLU(),
            nn.Linear(2 * width, 2 * width), nn.ReLU(),
            nn.Linear(2 * width, width), nn.ReLU(),
            nn.Linear(width, 1), nn.Sigmoid() if return_probs else nn.Sequential(),
        )

    def forward(self, z):
        return self.module(z)
