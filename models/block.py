import numpy as np

import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

'''
reference: https://github.com/saic-mdal/CIPS/blob/main/model/blocks.py
'''

def logistic_func(x, k, m):
    return torch.sigmoid(k * (x - m))

class ConLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True, freq_range=(0, 3)):
        super(ConLinear, self).__init__()
        self.linear = nn.Linear(ch_in, ch_out, bias=bias)
        if is_first:
            nn.init.uniform_(self.linear.weight[:ch_out//2], freq_range[0] / ch_in, freq_range[1] / ch_in)
            nn.init.uniform_(self.linear.weight[ch_out//2:], -freq_range[1] / ch_in, -freq_range[0] / ch_in)
        else:
            nn.init.uniform_(self.linear.weight, -np.sqrt(6. / ch_in), np.sqrt(6. / ch_in))

    def forward(self, x):
        return self.linear(x)


class SinActivation(nn.Module):
    def __init__(self,):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class LFF(nn.Module):
    def __init__(self, input_size, hidden_size, freq_range=(0, 3)):
        super(LFF, self).__init__()
        if type(freq_range) == int:
            freq_range = (0, freq_range)
        self.ffm = ConLinear(input_size, hidden_size, is_first=True, freq_range=freq_range)
        self.activation = SinActivation()

    def forward(self, x):
        x = self.ffm(x)
        x = self.activation(x)
        return x

    def get_freq_idx(self, freqs, ratio=False):
        freq_idx = []

        # Cin of weight should be 1
        weight = self.ffm.linear.weight.detach()[:, 0]
        weight_max = torch.max(weight)
        s = 0
        for f in freqs:
            if ratio:
                f = f * weight_max
            freq_idx.append(torch.logical_and(torch.abs(weight) >= s,
                                      torch.abs(weight) < f))
            s = f
        return freq_idx

    def get_freq_idx_logistic(self, freqs, ratio=False, k=3):
        freq_idx = []

        # Cin of weight should be 1
        weight = self.ffm.linear.weight.detach()[:, 0]
        prev_sum = 0
        if ratio:
            weight_max = torch.max(weight)
        for f in freqs:
            if ratio:
                f = f * weight_max
            f_ = 1 - logistic_func(torch.abs(weight), k=k, m=f)
            freq_idx.append(f_ - prev_sum)
            prev_sum = prev_sum + f_
        return freq_idx


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

class Blur3d(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        bs, c, t, h, w = input.size()
        input = input.view(bs, c * t, h, w)
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        _, _, h, w = out.size()
        out = out.view(bs, c, t, h, w) # pad
        return out

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out



class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out



class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate


    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

    def multiple_styles(self, input, styles, freq_idx):
        batch, in_channel, height, width = input.shape

        style_combined = torch.zeros((batch, 1, in_channel, 1, 1), dtype=torch.float32).cuda()

        for style, fidx in zip(styles, freq_idx):
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            style_combined = style_combined + style * fidx.view(1, 1, in_channel, 1, 1)

        style = style_combined

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
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

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

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

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out




class EqualConv3d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        # print(kernel_size)
        if type(kernel_size) is tuple:
            kernel_size_t, kernel_size_h, kernel_size_w = kernel_size[0], kernel_size[1], kernel_size[2]
        else:
            kernel_size_t, kernel_size_h, kernel_size_w = kernel_size, kernel_size, kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size_t, kernel_size_h, kernel_size_w)
        )
        # self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.scale = 1 / math.sqrt(in_channel * kernel_size_t * kernel_size_h * kernel_size_w)

        if type(stride) is tuple:
            self.stride = stride
        else:
            # self.stride = (1, stride, stride)
            self.stride = (stride, stride, stride)
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, input):

        if type(self.padding) is tuple:
            # print("pad")
            # print(input.size())
            input = F.pad(input, self.padding)
            # print(input.size())
            out = F.conv3d(
                input,
                self.weight * self.scale,
                bias=self.bias,
                stride=self.stride,
            )
        else:
            out = F.conv3d(
                input,
                self.weight * self.scale,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
            )
        return out

    # def __repr__(self):
    #     return (
    #         f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
    #         f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
    #     )

class Conv3dLayer(nn.Sequential):
    # Currently, do not use bilinear downsampling, which means do not apply blur kernels
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        downsample_t=False,
        blur_kernel=(1, 3, 3, 1),
        bias=True,
        activate=True,
        padding="zero",
    ):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur3d(blur_kernel, pad=(pad0, pad1)))
            # (0, 1, 0, 1) is only for 3x3 kernel

            if downsample_t:
                stride = (2, 2, 2)
                if kernel_size == 3:
                    self.padding = (0, 0, 0, 0, 0, 1) # only padding temporal dimension
                else:
                    self.padding = 0
            else:
                stride = (1, 2, 2)
                if kernel_size == 3:
                    self.padding = (0, 0, 0, 0, 1, 1)
                else:
                    self.padding = 0


            '''
            # original code
            stride=2
            if kernel_size == 3:
                self.padding = (0, 1, 0, 1, 0, 1)
            else:
                self.padding = 0
            '''
        else:
            stride = 1
            self.padding = (kernel_size - 1) // 2

        layers.append(
            EqualConv3d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class Conv2p1dLayer(nn.Sequential):
    # Currently, do not use bilinear downsampling, which means do not apply blur kernels
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=(1, 3, 3, 1),
        bias=True,
        activate=True,
        padding="zero",
    ):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            # p = (len(blur_kernel) - factor) + (kernel_size - 1)
            # pad0 = (p + 1) // 2
            # pad1 = p // 2

            # layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            # (0, 1, 0, 1) is only for 3x3 kernel
            if kernel_size == 3:
                # self.padding = (0, 1, 0, 1, 1, 1)
                self.padding2 = (0, 1, 0, 1, 0, 0)
                self.padding1 = (0, 0, 0, 0, 0, 1)
            else:
                self.padding2 = 0
                self.padding1 = 0
            stride = 2

        else:
        # if not downsample:
            if padding == "zero":
                # self.padding = (kernel_size - 1) // 2
                pad = (kernel_size - 1) // 2
                self.padding2 = (pad, pad, pad, pad, 0, 0)
                self.padding1 = (0, 0, 0, 0, pad, pad)
                self.padding = 0


            elif padding == "reflect":
                padding = (kernel_size - 1) // 2

                if padding > 0:
                    layers.append(nn.ReflectionPad2d(padding))

                self.padding = 0

            elif padding != "valid":
                raise ValueError('Padding should be "zero", "reflect", or "valid"')

        if kernel_size > 1:
            kernel_size2 = (1, kernel_size, kernel_size)
            kernel_size1 = (kernel_size, 1, 1)
            stride2 = (1, stride, stride)
            stride1 = (stride, 1, 1)

            # print(stride2, stride1)

            layers.append(
                EqualConv3d(
                    in_channel,
                    out_channel,
                    kernel_size2,
                    padding=self.padding2,
                    stride=stride2,
                    bias=bias and not activate,
                )
            )
            layers.append(
                EqualConv3d(
                    out_channel,
                    out_channel,
                    kernel_size1,
                    padding=self.padding1,
                    stride=stride1,
                    bias=bias and not activate,
                )
            )
        else:
            layers.append(
                EqualConv3d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class Res3dBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        downsample=False,
        downsample_t=False,
        upsample=False,
        padding="zero",
        blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        if downsample and upsample:
            print("Error: cannot use both downsample and upsample in ResBlock")
            exit(-1)

        self.conv1 = Conv3dLayer(in_channel, out_channel, 3, padding=padding)
        # self.conv1 = Conv2p1dLayer(in_channel, out_channel, 3, padding=padding)

        self.conv2 = Conv3dLayer(
        # self.conv2 = Conv2p1dLayer(
            out_channel,
            out_channel,
            3,
            downsample=downsample,
            downsample_t=downsample_t,
            upsample=upsample,
            padding=padding,
            blur_kernel=blur_kernel,
        )

        if downsample or upsample or in_channel != out_channel:
            self.skip = Conv3dLayer(
            # self.skip = Conv2p1dLayer(
                in_channel,
                out_channel,
                1,
                downsample=downsample,
                downsample_t=downsample_t,
                upsample=upsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False,
            )

        else:
            self.skip = None

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        # print(out.size(), skip.size())

        return (out + skip) / math.sqrt(2)

class ToRGB_nostyle(nn.Module):
    def __init__(self, in_channel, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        # self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.conv = ConvLayer(in_channel, 3, 1, bias=False, activate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, skip=None):
        out = self.conv(input)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class StyledLinear(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        style_dim,
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            1,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=demodulate,
        )

        # self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        input = input.unsqueeze(-1).unsqueeze(-1)
        out = self.conv(input, style)
        # out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out.squeeze(-1).squeeze(-1)

    def multiple_styles(self, input, styles, freq_idx):
        input = input.unsqueeze(-1).unsqueeze(-1)
        out = self.conv.multiple_styles(input, styles, freq_idx)
        # out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out.squeeze(-1).squeeze(-1)
