import os
import PIL

import functools

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models import discriminator, generator

from trainers import Trainer, accumulate

import data

import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim

def video_transform(video, image_transform):
    vid = []
    for im in video:
        vid.append(image_transform(im))

    vid = torch.stack(vid).permute(1, 0, 2, 3)

    return vid

import cfg

if __name__ == "__main__":
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    if args.ngpus > 1:
        args.distributed = True

    # for multi GPU
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.distributed.barrier()

    print(args)
    n_channels = 3

    if args.center_crop:
        # THIS IS FOR UCF101 (with original mp4 format)
        image_transforms = transforms.Compose([
            transforms.CenterCrop(240),
            transforms.Resize(int(args.image_size)),
            transforms.ToTensor(),
            lambda x: x[:n_channels, ::],
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        image_transforms = transforms.Compose([
            transforms.Resize(int(args.image_size)),
            transforms.ToTensor(),
            lambda x: x[:n_channels, ::],
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    video_length = int(args.video_length)
    image_batch = int(args.image_batch)
    video_batch = int(args.video_batch)

    dim_z_content = int(args.dim_z_content)
    dim_z_motion = int(args.dim_z_motion)
    size = int(args.image_size)

    freqs = []
    for f in args.freq_val.split(','):
        freqs.append(int(f))

    # Dataloader
    dataset = data.Dataset(args.data_path, args.video_length, args.every_nth, image_transforms, video_transforms, delta_t=(args.video_discriminator == "DeltaVideoDiscriminator"))

    # for multi GPU
    if args.distributed:
        dataset_sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(dataset, batch_size=max(image_batch, video_batch), drop_last=True, num_workers=16, sampler=dataset_sampler)
    else:
        loader = DataLoader(dataset, batch_size=max(image_batch, video_batch), drop_last=True, num_workers=16, shuffle=True)

    g = generator.VideoGenerator(size, dim_z_content, dim_z_motion, video_length, freqs=freqs)
    g_ema = generator.VideoGenerator(size, dim_z_content, dim_z_motion, video_length, freqs=freqs)
    
    video_size = int(args.image_size * args.VD_resize_ratio)
    image_discriminator = getattr(discriminator, args.image_discriminator)(size=args.image_size, args=args)
    video_discriminator = getattr(discriminator, args.video_discriminator)(size=video_size, args=args)

    g_ema.eval()
    
    # for multi GPU
    accumulate(g_ema, g, 0)

    if torch.cuda.is_available():
        g_ema.cuda()
        g.cuda()
        image_discriminator.cuda()
        video_discriminator.cuda()

    # create optimizers
    opt_generator = optim.Adam(g.parameters(), lr=0.002, betas=(0.0, 0.99))
    opt_image_discriminator = optim.Adam(image_discriminator.parameters(), lr=0.002, betas=(0.0, 0.99))
    opt_video_discriminator = optim.Adam(video_discriminator.parameters(), lr=0.002, betas=(0.0, 0.99))

    if args.ckpt_path is not None:
        print('load pretrained models...')
        ckpt = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
        g.load_state_dict(ckpt['g'])
        g_ema.load_state_dict(ckpt['g_ema'])
        image_discriminator.load_state_dict(ckpt['img_d'])
        video_discriminator.load_state_dict(ckpt['vid_d'])

        opt_generator.load_state_dict(ckpt['g_optim'])
        opt_image_discriminator.load_state_dict(ckpt['img_d_optim'])
        opt_video_discriminator.load_state_dict(ckpt['vid_d_optim'])

        cur_iter = ckpt['iter']
    else:
        cur_iter = 0

    # for multi GPU
    if args.distributed:
        g = nn.parallel.DistributedDataParallel(
            g,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        image_discriminator = nn.parallel.DistributedDataParallel(
            image_discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        video_discriminator = nn.parallel.DistributedDataParallel(
            video_discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    trainer = Trainer(loader,
                      int(args.print_every),
                      int(args.batches),
                      args.log_path,
                      use_cuda=torch.cuda.is_available(),
                      use_delta=(args.video_discriminator == "DeltaVideoDiscriminator"),
                      args=args)
    
    trainer.cur_iter = cur_iter
    trainer.train(g_ema, g, image_discriminator, video_discriminator,
                  opt_generator, opt_image_discriminator, opt_video_discriminator)
