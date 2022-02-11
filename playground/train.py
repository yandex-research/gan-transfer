import os
import sys
import argparse
import json
from copy import deepcopy
from enum import Enum, auto
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from playground.test import test, plot_samples, make_gif
from playground.model import Generator, Discriminator
from playground.data import make_datasampler

_checkpoint_base_name = 'state_dict'


class LossType(Enum):
    vanila = auto()
    js = auto()
    wasserstein = auto()
    hinge = auto()


class TrainParams:
    def __init__(self, **kwargs):
        self.steps = 5000
        self.batch_size = 64
        self.rate = 0.0002
        self.betas = (0.5, 0.999)

        self.disc_steps = 4
        self.top_k = False

        self.gradient_penalty = 0.0
        self.loss = LossType.vanila

        self.steps_per_img_save = 1000
        self.steps_per_log = 250
        self.steps_per_test = 2000
        self.steps_per_checkpoint = 1000
        self.steps_per_save = 500
        self.steps_per_checkpoint_global_save = 1000

        self.ema_alpha = 0.99

        for key, val in kwargs.items():
            if key == 'loss' and isinstance(val, str):
                val = getattr(LossType, val)
            if val is not None:
                self.__dict__[key] = val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, val in TrainParams().__dict__.items():
        target_type = type(val) if val is not None else int
        if target_type is LossType:
            target_type = str
        parser.add_argument('--{}'.format(key), type=target_type, default=None)
    parser.add_argument('--out', type=str, help='out directory')
    parser.add_argument('--size', type=int, default=4)
    parser.add_argument('--data', type=str,
                        choices=['gaussian_grid', 'gaussian_circle', 'grid_2d', 'spiral'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.random.manual_seed(args.seed)
    torch.cuda.set_device(args.device)

    with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
        command_file.write(' '.join(sys.argv))

    sampler = make_datasampler(args.data)
    generator = Generator()
    discriminator = Discriminator()
    params = TrainParams(**args.__dict__)
    print(f'train params: {params.__dict__}')
    train(generator, discriminator, sampler, args.out, params=params)


def blend(current, incoming, alpha_1=0.99, alpha_2=None):
    # handle dict-like or module inputs
    try:
        current_dict = current.state_dict()
        incoming_dict = incoming.state_dict()
    except Exception:
        current_dict = current
        incoming_dict = incoming

    if alpha_2 is None:
        alpha_2 = 1.0 - alpha_1

    for name in current_dict.keys():
        current_dict[name] = alpha_1 * current_dict[name] + alpha_2 * incoming_dict[name].data

    try:
        current.load_state_dict(current_dict)
    except Exception:
        current = current_dict

    return current


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device='cuda'):
    """Calculates the gradient penalty loss"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand([real_samples.size(0), 1], device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1.0 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def generator_loss(discriminator, fake, loss_type):
    if loss_type == LossType.vanila:
        return -(torch.log(discriminator(fake))).mean()
    elif loss_type == LossType.js:
        return (torch.log(1.0 - discriminator(fake))).mean()
    elif loss_type in [LossType.wasserstein, LossType.hinge]:
        return -(discriminator(fake)).mean()


def discriminator_loss(discriminator, real, fake, params: TrainParams):
    if params.loss in [LossType.js, LossType.vanila]:
        loss = -(torch.log(discriminator(real))).mean() - \
               (torch.log(1.0 - discriminator(fake))).mean()

    elif params.loss == LossType.wasserstein:
        real_validity = discriminator(real).mean()
        fake_validity = discriminator(fake).mean()
        loss = -real_validity + fake_validity

    elif params.loss == LossType.hinge:
        loss = F.relu(1.0 - discriminator(real)).mean() + F.relu(1.0 + discriminator(fake)).mean()

    if params.gradient_penalty > 0.0:
        loss += \
            params.gradient_penalty * compute_gradient_penalty(discriminator, real.data, fake.data)
    return loss


def train(generator, discriminator, sampler, out_dir, checkpoint=None, try_load=False,
          params=TrainParams()):
    imgs_dir = os.path.join(out_dir, 'images')
    tboard_dir = os.path.join(out_dir, 'tensorboard')
    checkpints_dir = os.path.join(out_dir, 'checkpoints')
    out_json = f'{out_dir}/metrics.json'

    for d in [imgs_dir, checkpints_dir, tboard_dir]: os.makedirs(d, exist_ok=True)
    writer = make_writer(generator, discriminator, tboard_dir)

    generator.train().cuda()
    discriminator.train().cuda()
    fixed_z = torch.randn([2**13, generator.dim_z])
    generator_ema = deepcopy(generator).eval()

    # optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=params.rate, betas=params.betas)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=params.rate, betas=params.betas)

    step = 0
    g_step = 0
    # Checkpoint load
    if (checkpoint is None or ~os.path.isfile(checkpoint)) and try_load:
        print(f'trying to load latest checkpoint from {checkpints_dir}')
        checkpoint = latest_checkpoint(checkpints_dir, _checkpoint_base_name)
    if checkpoint is not None:
        g_step = load_from_checkpoint(checkpoint, generator, generator_ema,
                                      discriminator, optimizer_G, optimizer_D)
        print(f'start from checkpoint at step {g_step}')

    # Actual training
    while True:
        # D step
        real_samples = sampler(params.batch_size).cuda()
        z = torch.randn([params.batch_size, generator.dim_z], device='cuda')
        fake_samples = generator(z)

        optimizer_D.zero_grad()
        d_loss = discriminator_loss(discriminator, real_samples, fake_samples, params)
        d_loss.backward()
        optimizer_D.step()

        # G step
        if step % params.disc_steps == 0:
            optimizer_G.zero_grad()
            fake_samples = generator(
                torch.randn([params.batch_size, generator.dim_z], device='cuda'))

            g_loss = generator_loss(discriminator, fake_samples, params.loss)
            g_loss.backward()
            optimizer_G.step()

            generator_ema = blend(generator_ema, generator.eval(), alpha_1=params.ema_alpha)
            generator.train()

            # Log and Test
            log(generator, generator_ema, discriminator, optimizer_G, optimizer_D,
                d_loss, g_loss, params, fixed_z, sampler.limits,
                writer, g_step, checkpints_dir, imgs_dir)
            if g_step % params.steps_per_test == 0:
                test(generator, sampler, imgs_dir, out_json, writer, 'W1_g', g_step)
                test(generator_ema, sampler, imgs_dir, out_json, writer, 'W1_g_ema', g_step)

            g_step += 1
        step += 1

        if g_step == params.steps:
            break

    save_checkpoint(f'{checkpints_dir}/models_final.pt',
                    generator, generator_ema, discriminator, g_step)
    make_gif(imgs_dir, 'gen')
    make_gif(imgs_dir, 'gen_ema')


@torch.no_grad()
def log(generator, generator_ema, discriminator, optimizer_G, optimizer_D, d_loss, g_loss, params,
        fixed_z, limits, writer, g_step, checkpints_dir, imgs_dir):
    if g_step % params.steps_per_log == 0:
        log_training(writer, params, g_step, d_loss, g_loss)

    if g_step % params.steps_per_img_save == 0:
        samples_g = plot_samples(generator, fixed_z,
                                 title=f'g | setp {g_step}', limits=limits)
        samples_g_ema = plot_samples(generator_ema, fixed_z,
                                     title=f'g-ema | setp {g_step}', limits=limits)

        samples_g.save(f'{imgs_dir}/gen_{str(g_step).zfill(6)}.jpg')
        samples_g_ema.save(f'{imgs_dir}/gen_ema_{str(g_step).zfill(6)}.jpg')

    if g_step % params.steps_per_save == 0 or g_step == params.steps:
        save_checkpoint(f'{checkpints_dir}/models_{g_step}.pt',
                        generator, generator_ema, discriminator, g_step)

    if g_step % params.steps_per_checkpoint == 0 and g_step > 0:
        save_checkpoint(f'{checkpints_dir}/{_checkpoint_base_name}.pt',
                        generator, generator_ema, discriminator, g_step,
                        optimizer_G, optimizer_D)

    if g_step % params.steps_per_checkpoint_global_save == 0 and g_step > 0:
        save_checkpoint(f'{checkpints_dir}/{_checkpoint_base_name}_{g_step}.pt',
                        generator, generator_ema, discriminator, g_step,
                        optimizer_G, optimizer_D)


def make_writer(generator, discriminator, tboard_dir):
    writer = SummaryWriter(tboard_dir)
    g_training = generator.training
    d_training = discriminator.training
    generator.eval()
    discriminator.eval()

    try:
        with torch.no_grad():
            z = torch.randn([1, generator.dim_z], device='cpu')
            writer.add_graph(generator, z)
            writer.add_graph(discriminator, generator(z))
    except Exception as e:
        print(f'failed to write graph: {e}')

    generator.train(g_training)
    discriminator.train(d_training)
    return writer


def latest_checkpoint(root_dir, base_name):
    checkpoints = [chkpt for chkpt in os.listdir(root_dir) if base_name in chkpt]
    if len(checkpoints) == 0:
        return None
    latest_chkpt = None
    latest_step = -1
    for chkpt in checkpoints:
        step = torch.load(f'{root_dir}/{chkpt}')['step']
        if step > latest_step:
            latest_chkpt = chkpt
            latest_step = step
    return f'{root_dir}/{latest_chkpt}'


def load_from_checkpoint(checkpoint, generator, generator_ema, discriminator,
                         optimizer_G, optimizer_D):
    data = torch.load(checkpoint, map_location='cpu')
    generator.load_state_dict(data['generator'])
    generator_ema.load_state_dict(data['generator_ema'])
    discriminator.load_state_dict(data['discriminator'])
    optimizer_G.load_state_dict(data['optimizer_G'])
    optimizer_D.load_state_dict(data['optimizer_D'])

    return data['step']


def save_checkpoint(checkpoint, generator, generator_ema, discriminator,
                    step, optimizer_G=None, optimizer_D=None):
    state_dict = {
        'step': step,
        'generator': generator.state_dict(),
        'generator_ema': generator_ema.state_dict(),
        'discriminator': discriminator.state_dict(),
    }
    if (optimizer_G is not None) and (optimizer_D is not None):
        state_dict.update({
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
        })

    torch.save(state_dict, checkpoint)


@torch.no_grad()
def log_training(writer, params, step, d_loss, g_loss):
    print(f'{int(100.0 * step / params.steps)}% | Step {step} :'
          f'D loss: {d_loss.item():0.3f} | G loss: {g_loss.item():0.3f}')

    writer.add_scalar('discriminator loss', d_loss.item(), step)
    writer.add_scalar('generator loss', g_loss.item(), step)


if __name__ == '__main__':
    main()
