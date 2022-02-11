import os
import io
import re
import json
import torch
import numpy as np
import ot
import ot.plot

from matplotlib import pyplot as plt
from PIL import Image
import imageio


@torch.no_grad()
def test(generator, sampler, imgs_dir, out_json, writer, title, step, count=2048):
    is_training = generator.training
    generator.eval()

    batch = 128
    real_samples = torch.cat([sampler(batch) for _ in torch.arange(0, count, batch)])
    gen_samples = torch.cat([
        generator(torch.randn(batch, generator.dim_z).cuda()) \
        for _ in torch.arange(0, count, batch)])

    w1, ot_img = w1_wassersein(real_samples.cpu().numpy(), gen_samples.cpu().numpy(), step)
    ot_img.save(f'{imgs_dir}/{title}_{step}.jpg')
    print(f'step {step} | {title} W1-loss: {w1:0.3f}')

    metrics = {}
    if os.path.isfile(out_json):
        with open(out_json, 'r') as f:
            metrics = json.load(f)
    try:
        metrics[str(step)][title] = w1
    except Exception:
        metrics[str(step)] = {title: w1}

    with open(out_json, 'w+') as f:
        json.dump(metrics, f)

    writer.add_scalar(title, w1, step)
    generator.train(is_training)


def w1_wassersein(samples_real, samples_gen, step):
    count = len(samples_real)
    M = ot.dist(samples_real, samples_gen, 'euclidean')
    G0 = ot.emd(np.ones(count) / count, np.ones(count) / count, M / M.max())

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes()
    ot.plot.plot2D_samples_mat(samples_real, samples_gen, G0, c=[.5, .5, 1])
    ax.plot(samples_real[:, 0], samples_real[:, 1], '+b', label='real')
    ax.plot(samples_gen[:, 0], samples_gen[:, 1], 'xr', label='generated')
    ax.legend(loc=0)
    ax.set_title(f'OT step {step}')

    return float(np.mean(M[np.nonzero(G0)])), fig_to_img(fig)


@torch.no_grad()
def plot_samples(g, z, batch=512, title='', limits=[-1, 4]):
    is_training = g.training
    g.eval()
    z = z.cuda()
    samples = []
    for start in torch.arange(0, len(z), batch):
        samples.append(g(z[start: start + batch]).cpu())
    samples = torch.cat(samples)
    g.train(is_training)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(xlim=limits, ylim=limits)
    ax.grid()
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.05)
    ax.set_title(title)
    return fig_to_img(fig)


def make_gif(root_dir, prefix):
    files = [f'{root_dir}/{f}' for f in os.listdir(root_dir) if \
             re.match(f"{prefix}_[0-9]*.jpg", f)]
    files.sort(key=lambda x: x.split('.')[0][-6:])
    imageio.mimsave(f'{root_dir}/{prefix}.gif', [Image.open(f) for f in files])


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
