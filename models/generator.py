import numpy as np

import torch.nn as nn
from models.block import *

FREQ_MAX = 1e+8

class VideoGenerator(nn.Module):
    def __init__(self, size, dim_z_content, dim_z_motion, video_length, n_motion_linear=2, n_mlp_m=2, n_mlp_c=8, freqs=[3, 5, 7]):
        super().__init__()

        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.video_len = video_length

        # Learned Fourier Features with disentangled frequency
        self.LFF = nn.ModuleList()
        freq_last = 0
        for f in freqs:
            if f == freq_last:
                freq_last = 0
            self.LFF.append(LFF(1, dim_z_motion, freq_range=(freq_last, f)))
            freq_last = f

        layers = [PixelNorm()]
        for i in range(n_mlp_m):
            layers.append(
                EqualLinear(
                    dim_z_motion, dim_z_motion, lr_mul=0.01, activation="fused_lrelu"
                )
            )
        # mapping network of motion generator
        self.motion_style = nn.Sequential(*layers)

        # motion generator
        self.motion_linear = nn.ModuleList()
        self.freqs = freqs
        for _ in freqs:
            out_dim = dim_z_motion
            temp_list = nn.ModuleList()
            for i in range(n_motion_linear):
                if i == n_motion_linear - 1:
                    out_dim = dim_z_content
                if i == 0:
                    # Modulate the first motion generator
                    temp_list.append(StyledLinear(dim_z_motion, out_dim, dim_z_motion, demodulate=False))
                else:
                    temp_list.append(EqualLinear(dim_z_motion, out_dim))
            self.motion_linear.append(temp_list)

        self.renderer = Generator(size, dim_z_content, n_mlp=n_mlp_c)


    def sample_rand(self, num_samples, grid_range=(-1, 1), on_grid=True, num_coords=1):
        if on_grid:
            coords_t = torch.linspace(grid_range[0], grid_range[1], self.video_len + 1)[:-1]
            coords = coords_t[torch.randint(len(coords_t), size=(num_samples, num_coords))]
        else:
            coords = torch.rand(size=[num_samples, 1], dtype=torch.float32)
            coords = coords * (grid_range[1] - grid_range[0]) + grid_range[0]
        return coords.cuda()

    def sample_grid(self, num_samples, t=None, grid_range=(-1, 1)):
        if t == None:
            t = self.video_len
        coords = torch.linspace(grid_range[0], grid_range[1], t + 1)[:-1]
        return coords.repeat(num_samples, 1).cuda()

    def sample_z_m(self, num_samples, t=None):
        z_m = torch.randn(num_samples, self.dim_z_motion, dtype=torch.float32).cuda()
        return z_m

    def sample_z_content(self, num_samples):
        return torch.randn(num_samples, self.dim_z_content, dtype=torch.float32).cuda()

    def sample_video_noise(self, num_samples, video_len):
        # noise should be consistent throughout video
        noises_repeated = []
        noises = self.renderer.make_noise_batch(num_samples)
        
        # repeating interleaved
        for n in noises:
            noise_repeated = torch.repeat_interleave(n, video_len, dim=0)
            noises_repeated.append(noise_repeated)
        return noises_repeated

    def sample_z_videos(self, num_samples, video_len=None, grid_range=(-1, 1), mixing_reg=False, delta_t=False):
        if video_len == None:
            video_len = self.video_len
            
        if delta_t:
            video_len = 2
            coords = torch.sort(self.sample_rand(num_samples=num_samples, num_coords=2), dim=1)[0]
        else:
            coords = self.sample_grid(num_samples, t=video_len, grid_range=grid_range)

        if mixing_reg:
            z_m = self.sample_z_m(num_samples * 2)
        else:
            z_m = self.sample_z_m(num_samples)
        z_c = self.sample_z_content(num_samples)

        z_m = self.motion_style(z_m)

        z_m = torch.cat([z_m.unsqueeze(1)] * video_len, dim=1)
        z_c = torch.cat([z_c.unsqueeze(1)] * video_len, dim=1)

        bs, t = coords.size()

        if mixing_reg:
            z_m = [z_m[:num_samples].view(bs * t, -1), z_m[num_samples:].view(bs * t, -1)]
        else:
            z_m = z_m.view(bs * t, -1)
        z_c = z_c.view(bs * t, -1)

        coords_flat = coords.view(-1, 1)
        ffs = []

        for lff, molins in zip(self.LFF, self.motion_linear):
            ff = lff(coords_flat)
            for midx, motion_linear in enumerate(molins):
                if midx == 0:
                    if mixing_reg:
                        mix_ratio = torch.rand(1).cuda()
                        freq_idx = lff.get_freq_idx(freqs=[mix_ratio, FREQ_MAX], ratio=True)
                        ff = motion_linear.multiple_styles(ff, z_m, freq_idx)
                    else:
                        ff = motion_linear(ff, z_m)
                else:
                    # ff = motion_linear(ff, z_m)
                    ff = motion_linear(ff)
            ffs.append(ff)

        w = self.renderer.get_latent(z_c)
        w_combined = w
        for ff in ffs:
            w_combined = w_combined + ff

        return w_combined, w, ffs, coords


    def sample_videos(self, num_samples, video_len=None, grid_range=(-1, 1), mixing_reg=False, delta_t=False):
        if video_len == None:
            video_len = self.video_len
            
        w_combined, w, ffs, coords = self.sample_z_videos(num_samples, video_len, grid_range, mixing_reg=mixing_reg, delta_t=delta_t)

        if delta_t:
            video_len = 2
        noises_repeated = self.sample_video_noise(num_samples, video_len)

        vid, _ = self.renderer([w_combined], noise=noises_repeated, input_is_latent=True)

        _, c, h, w = vid.size()
        vid = vid.view(num_samples, video_len, c, h, w).permute(0, 2, 1, 3, 4)

        delta = coords[:, 1] - coords[:, 0]
        return vid, delta

    def sample_videos_from_zs(self, num_samples, zs, video_len=None, grid_range=(-1, 1), video_noise=None):
        if video_len == None:
            video_len = self.video_len
        z_m, z_c = zs

        # if z_m is list, z_m is different for every freq. motion generator
        if type(z_m) == list:
            z_ms = z_m
        else:
            z_ms = [z_m] * len(self.freqs)

        _z_ms = z_ms
        z_ms = []
        for z_m in _z_ms:
            z_m = self.motion_style(z_m)
            
            # when motion code inputed with temporal dimension
            if len(z_m.size()) < 3:
                z_m = torch.cat([z_m.unsqueeze(1)] * video_len, dim=1)
                z_m = z_m.view(num_samples * video_len, -1)
                z_ms.append(z_m)
                
        z_c = torch.cat([z_c.unsqueeze(1)] * video_len, dim=1)
        z_c = z_c.view(num_samples * video_len, -1)

        coords = self.sample_grid(num_samples, t=video_len, grid_range=grid_range)
        coords_flat = coords.view(-1, 1)
        ffs = []

        for idx, (z_m, lff, molins) in enumerate(zip(z_ms, self.LFF, self.motion_linear)):
            ff = lff(coords_flat)
            for idxm, motion_linear in enumerate(molins):
                if idxm == 0:
                    ff = motion_linear(ff, z_m)
                else:
                    # ff = motion_linear(ff, z_m)
                    ff = motion_linear(ff)
            ffs.append(ff)

        w = self.renderer.get_latent(z_c)
        w_combined = w
        for ff in ffs:
            w_combined += ff

        if video_noise == None:
            noises_repeated = self.sample_video_noise(num_samples, video_len)
        else:
            noises_repeated = video_noise

        vid, _ = self.renderer([w_combined], noise=noises_repeated, input_is_latent=True)

        _, c, h, w = vid.size()
        vid = vid.view(num_samples, video_len, c, h, w).permute(0, 2, 1, 3, 4)

        return vid, None

    def sample_videos_from_zs_freqs_zeros(self, num_samples, zs, video_len=None, grid_range=(-1, 1), freqs=None, nonzeros=None, video_noise=None):
        if video_len == None:
            video_len = self.video_len
        z_m, z_c = zs
        if freqs == None:
            freqs = [1e+8]
            
        # if z_m is list, z_m is different for every freq. motion generator
        if type(z_m) == list:
            z_ms = z_m
        else:
            z_ms = [z_m] * len(freqs)

        _z_ms = z_ms
        z_ms = []

        for z_m in _z_ms:
            z_m = self.motion_style(z_m)
            
            # when motion code inputed with temporal dimension
            if len(z_m.size()) < 3:
                z_m = torch.cat([z_m.unsqueeze(1)] * video_len, dim=1)
                z_m = z_m.view(num_samples * video_len, -1)
                z_ms.append(z_m)
        z_c = torch.cat([z_c.unsqueeze(1)] * video_len, dim=1)
        z_c = z_c.view(num_samples * video_len, -1)

        coords = self.sample_grid(num_samples, t=video_len, grid_range=grid_range)
        coords_flat = coords.view(-1, 1)

        activations = []
        ffs = []
        for idx, (lff, molins) in enumerate(zip(self.LFF, self.motion_linear)):
            ff = lff(coords_flat)
            freq_idx = lff.get_freq_idx(freqs)

            for midx, motion_linear in enumerate(molins):
                if midx == 0:
                    if nonzeros != None:
                        ff_new = torch.zeros_like(ff)
                        for zidx in nonzeros:
                            ff_new += ff * freq_idx[zidx]
                        ff = ff_new
                    ff = motion_linear.multiple_styles(ff, z_ms, freq_idx)
                else:
                    ff = motion_linear(ff)
                activations.append(ff)

            ffs.append(ff)

        w = self.renderer.get_latent(z_c)
        w_combined = w
        for ff in ffs:
            w_combined += ff

        if video_noise == None:
            noises_repeated = self.sample_video_noise(num_samples, video_len)
        else:
            noises_repeated = video_noise

        vid, _ = self.renderer([w_combined], noise=noises_repeated, input_is_latent=True)

        _, c, h, w = vid.size()
        vid = vid.view(num_samples, video_len, c, h, w).permute(0, 2, 1, 3, 4)

        return vid, freq_idx

    def sample_images(self, num_samples, grid_range=(-1, 1), on_grid=True, mixing_reg=False):

        if mixing_reg:
            z_m = self.sample_z_m(num_samples * 2)
        else:
            z_m = self.sample_z_m(num_samples)
        z_c = self.sample_z_content(num_samples)

        z_m = self.motion_style(z_m)

        coords = self.sample_rand(num_samples, grid_range, on_grid=on_grid)
        coords_flat = coords
        ffs = []

        if mixing_reg:
            z_m = [z_m[:num_samples], z_m[num_samples:]]

        for lff, molins in zip(self.LFF, self.motion_linear):
            ff = lff(coords_flat)
            for midx, motion_linear in enumerate(molins):
                if midx == 0:
                    if mixing_reg:
                        mix_ratio = torch.rand(1).cuda()
                        freq_idx = lff.get_freq_idx(freqs=[mix_ratio, FREQ_MAX], ratio=True)
                        ff = motion_linear.multiple_styles(ff, z_m, freq_idx)
                    else:
                        ff = motion_linear(ff, z_m)
                else:
                    ff = motion_linear(ff)
            ffs.append(ff)

        w = self.renderer.get_latent(z_c)
        w_combined = w
        for ff in ffs:
            w_combined = w_combined + ff

        imgs, _ = self.renderer([w_combined], input_is_latent=True)

        return imgs, None

    def forward(self, num_samples_vid, num_samples_img, delta_t=False, video_len=None, grid_range=(-1, 1), on_grid=True, mixing_prob=0.0):
        mixing_reg = True if np.random.rand() < mixing_prob else False
        videos, delta = self.sample_videos(num_samples_vid, video_len, grid_range, mixing_reg=mixing_reg, delta_t=delta_t)
        images, _ = self.sample_images(num_samples_img, grid_range, on_grid, mixing_reg=mixing_reg)
        return videos, images, delta


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
                
        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer("noise_{}".format(layer_idx), torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def make_noise_batch(self, batch_size):
        device = self.input.input.device

        noises = [torch.randn(batch_size, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(batch_size, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                
                noise = [
                    getattr(self.noises, "noise_{}".format(i)) for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None