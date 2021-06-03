""" Module implementing GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""
import datetime
import os
import time
import timeit
import copy
import numpy as np
import torch as th

from torch.nn import ModuleList
from .msggan_layers import DisGeneralConvBlock, DisFinalBlock, EqualConv2d
from torch.nn import Conv2d


from torch.nn import ModuleList, Conv2d
from .msggan_layers import GenGeneralConvBlock, PixelNorm, EqualConv2d, EqualLinear


class Generator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, use_eql=True, n_mlp=8, base_channel=512):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        

        super().__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size
        self.base_channel = base_channel

        # n_MLP
        layers = [PixelNorm()]
        for i in range(n_mlp):
            if self.use_eql:
                layers.append(EqualLinear(latent_size, latent_size))
            else:
                layers.append(th.nn.Linear(latent_size, latent_size))
            layers.append(th.nn.LeakyReLU(0.2))
        self.style = th.nn.Sequential(*layers)

        # register the modules required for the Generator Below ...
        # create the ToRGB layers for various outputs:
        if self.use_eql:
            def to_rgb(in_channels):
                return EqualConv2d(in_channels, 3, (1, 1), bias=True)
        else:
            def to_rgb(in_channels):
                return Conv2d(in_channels, 3, (1, 1), bias=True)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([GenGeneralConvBlock(in_channels=base_channel, out_channels=base_channel,
                                                      style_dim=latent_size, fused=False, initial=True)])
        self.rgb_converters = ModuleList([to_rgb(base_channel)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(in_channels=base_channel, out_channels=base_channel, style_dim=latent_size,
                                            fused=False, initial=False)
                rgb = to_rgb(base_channel)
            else:
                layer = GenGeneralConvBlock(int(base_channel // np.power(2, i - 3)),
                                            int(base_channel // np.power(2, i - 2)),
                                            style_dim=latent_size, fused=False, initial=False)
                rgb = to_rgb(int(base_channel // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

    def forward(self, x):
        """
        forward pass of the Generator
        :param x: input noise
        :return: *y => output of the generator at various scales
        """
        batch = x.size(0)
        style = self.style(x)
        noise = []
        for i in range(self.depth):
            size = 4 * 2 ** i
            noise.append(th.randn(batch, 1, size, size, device=style.device))
        outputs = []  # initialize to empty list
        outputs2 = []
        y = style  # start the computational pipeline
        for i, (block, converter) in enumerate(zip(self.layers, self.rgb_converters)):
            y = block(y, style, noise[i])
            outputs.append(converter(y))
            outputs2.append(y)

        return outputs,outputs2

    @staticmethod
    def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
        """
        adjust the dynamic colour range of the given input data
        :param data: input image data
        :param drange_in: original range of input
        :param drange_out: required range of output
        :return: img => colour range adjusted images
        """
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return th.clamp(data, min=0, max=1)

