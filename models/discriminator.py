import numpy as np

import torch.nn as nn
from models.block import *

# ==================================================
# Available models
# Image discriminators: PatchImageDiscriminator, ImageDiscriminator (default) 
# Video discriminators:PatchVideoDiscriminator, VideoDiscriminator (default), DeltaVideoDiscriminator
# ==================================================

class VideoDiscriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=(1, 3, 3, 1), args={}):
        super().__init__()

        channels = {
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

        convs = [Conv3dLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for idx, i in enumerate(range(log_size, 2, -1)):
            # since, video length is 16, only downsample twice (16 to 4)
            if idx < 2:
                downsample_t = True
            else:
                downsample_t = False

            out_channel = channels[2 ** (i - 1)]
            convs.append(Res3dBlock(in_channel, out_channel, downsample=True, downsample_t=downsample_t))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = Conv3dLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, input):
        out = self.convs(input)

        batch, channel, T, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, T, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4, 5], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, T, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        
        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)

        return out


class PatchVideoDiscriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=(1, 3, 3, 1), args={}):
        super().__init__()

        channels = {
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

        convs = [Conv3dLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for idx, i in enumerate(range(log_size, 2, -1)):
            # since, video length is 16, only downsample twice (16 to 4)
            if idx < 2:
                downsample_t = True
            else:
                downsample_t = False
            out_channel = channels[2 ** (i - 1)]

            convs.append(Res3dBlock(in_channel, out_channel, downsample=True, downsample_t=downsample_t))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = Conv3dLayer(in_channel, 1, 3)

    def forward(self, input):
        
        out = self.convs(input)
        out = self.final_conv(out)
        out = torch.mean(out, dim=[1, 2, 3, 4])

        return out


class DeltaVideoDiscriminator(nn.Module):
    def __init__(self, size, num_ts=2, channel_multiplier=2, blur_kernel=(1, 3, 3, 1), args={}):
        super().__init__()

        channels = {
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

        self.cmap_dim = 128 # hard-coded
        
        convs = [ConvLayer(3 * num_ts, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], self.cmap_dim),
        )

        self.temporalPE = LFF(1, 256) # 256 is hard-coded
        self.delta_linear = EqualLinear(256, self.cmap_dim) # 256 is hard-coded

        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, input, delta):
        
        assert len(input.shape) == 5, "Input shape should contain temporal dimension (Given shape: {})".format(input.shape)

        # Concat temporal images with channel dimension (input: [B, C, T, H, W])
        b, c, t, h, w = input.shape
        
        assert t == 2, "Currently, only support two images for video"
        input = input. reshape(b, c * t, h, w)

        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(out.shape[0], -1)
        
        # Conditional discriminator
        delta = self.delta_linear(self.temporalPE(delta))
        out = (self.final_linear(out) * delta).sum(dim=1, keepdim=True) / np.sqrt(self.cmap_dim)

        return out



class ImageDiscriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=(1, 3, 3, 1), args={}):
        super().__init__()

        channels = {
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
        
        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, input):

        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(out.shape[0], -1)
        
        out = self.final_linear(out)

        return out


class PatchImageDiscriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=(1, 3, 3, 1), args={}):
        super().__init__()

        channels = {
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
        
        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, 1, 3)


    def forward(self, input):

        out = self.convs(input)
        out = self.final_conv(out)
        
        out = torch.mean(out, dim=[1, 2, 3])

        return out