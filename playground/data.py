import inspect
from itertools import product
from dataclasses import dataclass
import numpy as np
import torch
from matplotlib import pyplot as plt

SAMPLES_DIM = 2


def make_grid(grid_size):
    grid_size = float(grid_size)
    return torch.tensor(list(product(
        *[torch.arange(grid_size) for _ in range(SAMPLES_DIM)])))


@dataclass
class GaussianMix():
    centers: torch.tensor = make_grid(4)
    sigma: float = 0.2
    size: float = 1.0

    def __post_init__(self):
        if isinstance(self.centers, str):
            if self.centers == 'grid':
                self.centers = make_grid(self.size)
            elif self.centers == 'circle':
                centers = self.size * np.roots([1] + [0] * 9 + [-1])
                self.centers = torch.from_numpy(np.stack([centers.real, centers.imag]).T)

        self.limits = [self.centers.min() - 1, self.centers.max() + 1]

    def __call__(self, batch_size):
        samples = torch.empty([batch_size, SAMPLES_DIM])
        for i in range(batch_size):
            target = torch.randint(len(self.centers), [])
            samples[i] = self.sigma * torch.randn([SAMPLES_DIM]) + self.centers[target]
        return samples


@dataclass
class Grid2d():
    size: int = 3

    def __post_init__(self):
        self.limits = [-1, self.size + 1]

    def __call__(self, batch_size):
        samples = torch.randint(0, self.size + 1, [batch_size, 2]).to(torch.float)
        local_shift = torch.rand([batch_size])
        xy_mask = torch.randint(0, 2, [batch_size]).to(torch.bool)
        samples[xy_mask, 0] += local_shift[xy_mask]
        samples[~xy_mask, 1] += local_shift[~xy_mask]

        return samples


@dataclass
class Spiral():
    num_rotations: int = 5
    r_step: float = 0.2
    r_0: float = 1

    def __post_init__(self):
        max_r = self.r_0 + self.r_step * self.num_rotations
        self.limits = [-max_r, max_r]

    def __call__(self, batch_size):
        alpha = torch.rand([batch_size, 1])
        angle = 2 * np.pi * (alpha * self.num_rotations)
        r = self.r_0 + (alpha * self.num_rotations * self.r_step)
        direction = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        return r * direction


@dataclass
class FatCircle():
    r: float
    sigma: float
    center: torch.tensor = torch.tensor((0.0, 0.0))

    def __post_init__(self):
        self.limits = [-self.r - 3 * self.sigma, self.r + 3 * self.sigma]

    def __call__(self, batch_size):
        angle = 2 * np.pi * torch.rand([batch_size, 1])
        direction = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        offset = torch.randn([batch_size]).view(-1, 1) * direction * self.sigma
        return self.r * direction + offset + self.center


def filter_kwargs(target_method, kwargs):
    return {
        key: kwargs[key] for key in inspect.signature(target_method).parameters.keys()\
        if key in kwargs
    }


def plot_distribution(distribution, count=1000, axs=None, alpha=0.25, **kwargs):
    if axs is None:
        axs = plt.gca()
        axs.axis('equal')
    samples = distribution(count)
    axs.scatter(x=samples[:, 0], y=samples[:, 1], alpha=alpha, **kwargs)


def make_datasampler(data_type: str, **kwargs):
    if data_type == 'gaussian_grid':
        return GaussianMix(centers='grid', **filter_kwargs(GaussianMix.__init__, kwargs))
    elif data_type == 'gaussian_circle':
        return GaussianMix(centers='circle', **filter_kwargs(GaussianMix.__init__, kwargs))
    elif data_type == 'grid_2d':
        return Grid2d(**filter_kwargs(Grid2d.__init__, kwargs))
    elif data_type == 'spiral':
        return Spiral(**filter_kwargs(Spiral.__init__, kwargs))
