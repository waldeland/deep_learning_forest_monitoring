import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """`UNet` class is based on https://arxiv.org/abs/1505.04597"""

    def __init__(
        self,
        n_classes: int = 2,
        in_channels: int = 1,
        depth: int = 5,
        start_filts: int = 32,
        up_mode: str = "bilinear",
        merge_mode: str = "concat",
        partial_conv: bool = True,
        use_bn: bool = True,
        activation_func: callable = F.leaky_relu,
    ):
        """

        Args:
            n_classes: Number of output channels
            in_channels: Number of input channels
            depth: Depth of UNet = number of downsampling/upsampling-steps
            start_filts: Number of how many channels the first conv should produce (is doubled at each subsampling-step)
            up_mode : 'transpose' (convolution), 'bilinear' (interpolation), 'nearest' (interpolation)
            merge_mode: How to merge do skip-connection from encoder to decoder, 'none', 'concat' or 'add'
            partial_conv: Use partial-conv instead of normal convs (https://github.com/NVIDIA/partialconv)
            use_bn: Use batch-norm after convs before activation
            activation_func: Activation function
            dropout: Adds dropout with p=dropout in the decoder (if dropout>0)
            dropconnect: Adds dropconnect with p=dropconnect in the decoder (if dropconnect>0)
            bayesian_conv: Adds bayesian weights in the decoder
        """
        super(UNet, self).__init__()

        self.fow = [
            2 ** (depth + 2),
            2 ** (depth + 2),
        ]  # Minimum recommended input size ( 2**(depth-1) will also work, but is then dominated by edge effects)
        self.pad = [0, 0]
        self.valid = False

        assert up_mode in ("transpose", "bilinear", "nearest")
        assert merge_mode in ("concat", "add", "none")
        assert 'up_mode "upsample" is incompatible ith merge_mode "add"', not (
            self.up_mode == "upsample" and self.merge_mode == "add"
        )

        self.up_mode = up_mode
        self.merge_mode = merge_mode
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.n_classes = n_classes

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = _DownConv(
                ins,
                outs,
                pooling=pooling,
                use_bn=use_bn,
                partial_conv=partial_conv,
                activation_func=activation_func,
            )
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = _UpConv(
                ins,
                outs,
                up_mode=up_mode,
                merge_mode=merge_mode,
                partial_conv=partial_conv,
                activation_func=activation_func,
            )
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.Sequential(*self.down_convs)
        self.up_convs = nn.Sequential(*self.up_convs)
        self.conv_final = _conv1x1(outs, n_classes)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

            pass

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)
            if i == len(self.up_convs) - 1:
                x = self.conv_final(x)

        return x

    def get_valid_loss(self, loss):
        """
        Compute loss only for center part of image crop (to avoid edge effects)
        Args:
            loss (torch.nn.Loss): loss-function

        Returns:
            torch.Tensor

        """

        class Loss(nn.Module):
            def __init__(self, loss, p):
                super(Loss, self).__init__()

                self.loss = loss
                self.p = p

            def forward(self, x, y, weight=None):
                p = self.p
                if weight is None:
                    return self.loss._forward(x[:, :, p:-p, p:-p], y[:, :, p:-p, p:-p])

                else:
                    return self.loss._forward(
                        x[:, :, p:-p, p:-p],
                        y[:, :, p:-p, p:-p],
                        weight[:, :, p:-p, p:-p],
                    )

        return Loss(loss, 2 ** (self.depth + 2) // 2)


def _conv3x3(
    in_channels,
    out_channels,
    stride=1,
    padding=1,
    bias=True,
    groups=1,
    partial_conv=False,
    bayesian=False,
    dropconnect=0.0,
):
    if partial_conv:
        return PartialConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
    else:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )


def _upconv2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    elif mode == "nearest":
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode="nearest", scale_factor=2, align_corners=False),
            _conv1x1(in_channels, out_channels),
        )

    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
            _conv1x1(in_channels, out_channels),
        )


def _conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class _DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        pooling=True,
        use_bn=False,
        partial_conv=False,
        activation_func=F.relu,
    ):
        super(_DownConv, self).__init__()

        self.activation_func = activation_func
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = _conv3x3(
            self.in_channels,
            self.out_channels,
            bias=not (use_bn),
            partial_conv=partial_conv,
        )
        self.conv2 = _conv3x3(
            self.out_channels,
            self.out_channels,
            bias=not (use_bn),
            partial_conv=partial_conv,
        )

        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.bn2 = nn.BatchNorm2d(self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.use_bn:
            x = self.activation_func(self.bn1(self.conv1(x)))
            x = self.activation_func(self.bn2(self.conv2(x)))
        else:
            x = self.activation_func(self.conv1(x))
            x = self.activation_func(self.conv2(x))

        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class _UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        merge_mode="concat",
        up_mode="transpose",
        partial_conv=False,
        use_bn=False,
        activation_func=F.relu,
        dropout=0.0,
        dropconnect=0.0,
        bayesian_conv=False,
    ):
        super(_UpConv, self).__init__()

        self.dropout = dropout
        self.activation_func = activation_func
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = _upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == "concat":
            self.conv1 = _conv3x3(
                2 * self.out_channels,
                self.out_channels,
                dropconnect=dropconnect,
                bayesian=bayesian_conv,
                partial_conv=partial_conv,
            )
        else:
            # num of input channels to conv2 is same
            self.conv1 = _conv3x3(
                self.out_channels,
                self.out_channels,
                dropconnect=dropconnect,
                bayesian=bayesian_conv,
                partial_conv=partial_conv,
            )
        self.conv2 = _conv3x3(
            self.out_channels, self.out_channels, partial_conv=partial_conv
        )

        self.use_bn = use_bn

        if use_bn:
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.bn2 = nn.BatchNorm2d(self.out_channels)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout2d(dropout)

    def forward(self, from_down, from_up):
        """Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        elif self.merge_mode == "none":
            x = from_up
        elif self.merge_mode == "add":
            x = from_up + from_down
        else:
            raise NotImplementedError(self.merge_mode)

        x = self.activation_func(self.conv1(x))
        if self.use_bn:
            x = self.bn1(x)

        if self.dropout:
            x = self.dropout_layer(x)

        x = self.activation_func(self.conv2(x))
        if self.use_bn:
            x = self.bn2(x)

        return x

###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable


class PartialConv2d(nn.Conv2d):
    """ Implementation of Partial-convolution (https://github.com/NVIDIA/partialconv)"""
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output
