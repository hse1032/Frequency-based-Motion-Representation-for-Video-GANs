import os
import time

import numpy as np

from logger import Logger

import torch
import torch.nn.functional as F
from torch import nn, autograd

from torch.autograd import Variable

from DiffAugment_pytorch import DiffAugment, DiffAugment_vid

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(0, 2, 3, 1)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def videos_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(0, 1, 2, 3, 4)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def one_hot_to_class(tensor):
    a, b = np.nonzero(tensor)
    return np.unique(b).astype(np.int32)

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


class Trainer(object):
    def __init__(self, video_sampler, log_interval, train_batches, log_folder, use_cuda=False, use_delta=False, args=None):

        if args is not None:
            self.args = args

        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.category_criterion = nn.CrossEntropyLoss()

        self.video_sampler = video_sampler

        self.video_batch_size = self.args.video_batch
        self.image_batch_size = self.args.image_batch

        self.log_interval = log_interval
        self.train_batches = train_batches

        self.log_folder = log_folder

        self.use_cuda = use_cuda
        self.use_delta = use_delta

        self.video_enumerator = None

        self.cur_iter = 0

        if args.DiffAugment:
            self.DiffAug_policy = 'color,translation,cutout,flip'

    @staticmethod
    def ones_like(tensor, val=1.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    @staticmethod
    def zeros_like(tensor, val=0.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    def d_logistic_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def d_r1_loss(self, real_pred, real_img):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty

    def g_nonsaturating_loss(self, fake_pred):
        loss = F.softplus(-fake_pred).mean()

        return loss

    def sample_real_video_batch(self):
        if self.video_enumerator is None:
            self.video_enumerator = enumerate(self.video_sampler)

        batch_idx, batch = next(self.video_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.items():
                b[k] = v.cuda()

        if batch_idx == len(self.video_sampler) - 1:
            self.video_enumerator = enumerate(self.video_sampler)

        return b

    def train_discriminator(self, discriminator, batch, fake_batch, opt, delta=None):
        opt.zero_grad()

        # real / fakes
        if self.use_delta and delta is not None:
            real_labels = discriminator(batch, delta[0]) # real
            fake_labels = discriminator(fake_batch.detach(), delta[1]) # fake
        else:
            real_labels = discriminator(batch)
            fake_labels = discriminator(fake_batch.detach())

        l_discriminator = self.d_logistic_loss(real_labels, fake_labels)
        l_discriminator_final = l_discriminator

        l_discriminator_final.backward()
        opt.step()

        if self.cur_iter % self.args.d_reg_every == 0:
            batch.requires_grad = True
            if self.use_delta and delta is not None:
                real_pred = discriminator(batch, delta[0])
            else:
                real_pred = discriminator(batch)
            r1_loss = self.d_r1_loss(real_pred, batch)

            discriminator.zero_grad()
            (self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]).backward()
            opt.step()
        return l_discriminator

    def train_generator(self,
                        image_discriminator, video_discriminator,
                        fake_image_batch, fake_video_batch,
                        opt, delta=None):

        opt.zero_grad()

        # train on images
        fake_labels = image_discriminator(fake_image_batch)
        l_generator = self.g_nonsaturating_loss(fake_labels)

        # train on videos
        if self.use_delta:
            fake_labels = video_discriminator(fake_video_batch, delta)
        else:
            fake_labels = video_discriminator(fake_video_batch)
        
        l_generator += self.g_nonsaturating_loss(fake_labels)

        l_generator_final = l_generator

        l_generator_final.backward()
        opt.step()

        return l_generator

    def train(self, g_ema, generator, image_discriminator, video_discriminator,
              opt_generator, opt_image_discriminator, opt_video_discriminator):
        if self.use_cuda:
            g_ema.cuda()
            generator.cuda()
            image_discriminator.cuda()
            video_discriminator.cuda()

        # for multi GPU
        if self.args.distributed:
            g_module = generator.module
            id_module = image_discriminator.module
            vd_module = video_discriminator.module
        else:
            g_module = generator
            id_module = image_discriminator
            vd_module = video_discriminator

        # batch_size 32, ema_kimg 10 is DEFAULT SETTING OF STYLEGAN in FFHQ
        accum = 0.5 ** (max(self.image_batch_size, self.video_batch_size) * self.args.ngpus / (self.args.ema_kimg * 1000))

        if self.args.local_rank == 0:
            logger = Logger(self.log_folder)

        batch_num = self.cur_iter

        # training loop
        def sample_fake_image_batch(g, batch_size, on_grid=True):
            return g.sample_images(batch_size, on_grid=on_grid)

        def sample_fake_video_batch(g, batch_size):
            return g.sample_videos(batch_size)

        def init_logs():
            return {'l_gen': 0, 'l_image_dis': 0, 'l_video_dis': 0}

        logs = init_logs()
        start_time = time.time()

        kernel_size = int(1 / self.args.VD_resize_ratio)
        
        while True:
            generator.train()
            image_discriminator.train()
            video_discriminator.train()

            opt_generator.zero_grad()

            opt_video_discriminator.zero_grad()

            batch = self.sample_real_video_batch()
            real_image_batch = batch['images'][:self.image_batch_size]
            real_video_batch = batch['videos'][:self.video_batch_size]
            real_delta = batch['delta'][:self.video_batch_size]

            fake_video_batch, fake_image_batch, fake_delta = generator(self.video_batch_size, self.image_batch_size, mixing_prob=self.args.mixing_prob, delta_t=True)

            if self.args.VD_resize_ratio < 1.0:
                real_video_batch = F.avg_pool3d(real_video_batch, (1, kernel_size, kernel_size))
                fake_video_batch = F.avg_pool3d(fake_video_batch, (1, kernel_size, kernel_size))

            if self.args.DiffAugment:
                # Use all default augmentations
                real_image_batch = DiffAugment(real_image_batch, policy=self.DiffAug_policy)
                fake_image_batch = DiffAugment(fake_image_batch, policy=self.DiffAug_policy)
                real_video_batch = DiffAugment_vid(real_video_batch, policy=self.DiffAug_policy)
                fake_video_batch = DiffAugment_vid(fake_video_batch, policy=self.DiffAug_policy)

            # train image discriminator
            l_image_dis = self.train_discriminator(image_discriminator, real_image_batch,
                                                   fake_image_batch, opt_image_discriminator)

            # train video discriminator
            l_video_dis = self.train_discriminator(video_discriminator, real_video_batch,
                                                   fake_video_batch, opt_video_discriminator,
                                                   delta=[real_delta, fake_delta])

            # train generator
            l_gen = self.train_generator(image_discriminator, video_discriminator,
                                         fake_image_batch, fake_video_batch,
                                         opt_generator, delta=fake_delta)

            # for accumulate g_ema
            accumulate(g_ema, g_module, accum)

            logs['l_gen'] += l_gen.data
            logs['l_image_dis'] += l_image_dis.data
            logs['l_video_dis'] += l_video_dis.data

            batch_num += 1
            self.cur_iter += 1

            # for multi GPU
            if self.args.local_rank == 0:
                if batch_num % self.log_interval == 0:

                    log_string = "Batch %d" % batch_num
                    for k, v in logs.items():
                        log_string += " [%s] %5.3f" % (k, v / self.log_interval)

                    log_string += ". Took %5.2f" % (time.time() - start_time)

                    print(log_string)

                    for tag, value in logs.items():
                        logger.scalar_summary(tag, value / self.log_interval, batch_num)

                    logs = init_logs()
                    start_time = time.time()

                    with torch.no_grad():
                        g_ema.eval()

                        images, _ = sample_fake_image_batch(g_ema, self.args.num_print_images)
                        logger.image_summary("Images", images_to_numpy(images), batch_num)

                        videos, _ = sample_fake_video_batch(g_ema, self.args.num_print_images)
                        logger.video_summary("Videos", videos_to_numpy(videos), batch_num)

                        for idx, lff in enumerate(g_module.LFF):
                            logger.histogram_summary("LFF_frequency_{}".format(idx), lff.ffm.linear.weight, batch_num)
                            logger.histogram_summary("LFF_phase_{}".format(idx), lff.ffm.linear.bias, batch_num)

                        images_rand, _ = sample_fake_image_batch(g_ema, self.args.num_print_images, on_grid=False)
                        logger.image_summary("Images_rand", images_to_numpy(images_rand), batch_num)

                    if batch_num % (self.args.save_interval) == 0:
                        torch.save(
                            {
                                "g": g_module.state_dict(),
                                "g_ema": g_ema.state_dict(),
                                "img_d": id_module.state_dict(),
                                "vid_d": vd_module.state_dict(),
                                "g_optim": opt_generator.state_dict(),
                                "img_d_optim": opt_image_discriminator.state_dict(),
                                "vid_d_optim": opt_video_discriminator.state_dict(),
                                "args": self.args,
                                "iter": self.cur_iter,
                            },
                            os.path.join(self.log_folder, 'checkpoint_%06d.pt' % batch_num),
                        )
                    # save generator more densely
                    if batch_num % (self.args.G_save_interval) == 0:
                        torch.save(
                            {
                                "g_ema": g_ema.state_dict(),
                                "args": self.args,
                                "iter": self.cur_iter,
                            },
                            os.path.join(self.log_folder, 'G_ema_%06d.pt' % batch_num)
                        )
                    # this is for continue training
                    if batch_num % 1000 == 0:
                        torch.save(
                            {
                                "g": g_module.state_dict(),
                                "g_ema": g_ema.state_dict(),
                                "img_d": id_module.state_dict(),
                                "vid_d": vd_module.state_dict(),
                                "g_optim": opt_generator.state_dict(),
                                "img_d_optim": opt_image_discriminator.state_dict(),
                                "vid_d_optim": opt_video_discriminator.state_dict(),
                                "args": self.args,
                                "iter": self.cur_iter,
                            },
                            os.path.join(self.log_folder, 'checkpoint_latest.pt'),
                        )

                if batch_num >= self.train_batches:
                    torch.save(
                        {
                            "g": g_module.state_dict(),
                            "g_ema": g_ema.state_dict(),
                            "img_d": id_module.state_dict(),
                            "vid_d": vd_module.state_dict(),
                            "g_optim": opt_generator.state_dict(),
                            "img_d_optim": opt_image_discriminator.state_dict(),
                            "vid_d_optim": opt_video_discriminator.state_dict(),
                            "args": self.args,
                            "iter": self.cur_iter,
                        },
                        os.path.join(self.log_folder, 'checkpoint_%06d.pt' % batch_num),
                    )
                    break
