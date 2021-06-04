import datetime
import os
import time
import timeit
import copy
import numpy as np
import torch 

from torch.nn import ModuleList
from .msggan_layers import DisGeneralConvBlock, DisFinalBlock, EqualConv2d
from torch.nn import Conv2d


from torch.nn import ModuleList, Conv2d
from .msggan_layers import GenGeneralConvBlock, PixelNorm, EqualConv2d, EqualLinear


class Discriminator(torch.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, depth=7, feature_size=512,
                 use_eql=True, gpu_parallelize=False):
        """
        constructor for the class
        :param depth: total depth of the discriminator
                       (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use the equalized learning rate or not
        :param gpu_parallelize: whether to use DataParallel on the discriminator
                                Note that the Last block contains StdDev layer
                                So, it is not parallelized.
        """
        

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        # self.gpu_parallelize = gpu_parallelize
        self.use_eql = use_eql
        self.depth = depth
        self.feature_size = feature_size

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            def from_rgb(out_channels):
                return EqualConv2d(3, out_channels, (1, 1), bias=True)
        else:
            def from_rgb(out_channels):
                return Conv2d(3, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList()
        self.final_converter = from_rgb(self.feature_size // 2)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList()
        self.final_block = DisFinalBlock(self.feature_size, use_eql=self.use_eql)




        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = from_rgb(int(self.feature_size // np.power(2, i - 1)))
            else:
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size // 2,
                                            use_eql=self.use_eql)
                rgb = from_rgb(self.feature_size // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 2] = \
            from_rgb(self.feature_size // np.power(2, i - 2))

        # parallelize the modules from the module-lists if asked to:
        # if self.gpu_parallelize:
        #     for i in range(len(self.layers)):
        #         self.layers[i] = torch.nn.DataParallel(self.layers[i])
        #         self.rgb_to_features[i] = torch.nn.DataParallel(
        #             self.rgb_to_features[i])

        # Note that since the FinalBlock contains the StdDev layer,
        # it cannot be parallelized so easily. It will have to be parallelized
        # from the Lower level (from CustomLayers). This much parallelism
        # seems enough for me.

        print("DISCRIMINATOE Depth : ",self.depth, " feature_size : ",self.feature_size)
        print("RGB TO FEATURES ",self.rgb_to_features, "\n")
        print("DISC LAYERS : ",self.layers)

    def forward(self, inputs):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """

        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales"

        y = self.rgb_to_features[self.depth - 2](inputs[self.depth - 1])
        #print("Y.shape First",y.shape)
        
        y = self.layers[self.depth - 2](y)
        #print("Y.shape Second",y.shape)
        for x, block, converter in \
                zip(reversed(inputs[1:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = torch.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        # calculate the final block:
        input_part = self.final_converter(inputs[0])
        y = torch.cat((input_part, y), dim=1)
        y = self.final_block(y)

        # return calculated y
        return y


