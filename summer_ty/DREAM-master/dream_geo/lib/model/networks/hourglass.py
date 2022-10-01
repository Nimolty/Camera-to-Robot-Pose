# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:43:55 2022

@author: lenovo
"""
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as tviz_models

# Based on DopeHourglassBlockSmall, not using skipped connections
class DreamHourglass(nn.Module):
    def __init__(
        self,
        n_keypoints,
        n_image_input_channels=3,
        internalize_spatial_softmax=False,
        learned_beta=True,
        initial_beta=1.0,
        skip_connections=False,
        deconv_decoder=False,
        full_output=False,
    ):
        super(DreamHourglass, self).__init__()
        self.n_keypoints = n_keypoints
        self.n_image_input_channels = n_image_input_channels
        self.internalize_spatial_softmax = internalize_spatial_softmax
        self.skip_connections = skip_connections
        self.deconv_decoder = deconv_decoder
        self.full_output = full_output

        if 1:
            self.n_output_heads = 2
            self.learned_beta = learned_beta
            self.initial_beta = initial_beta
        else:
            self.n_output_heads = 1
            self.learned_beta = False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        vgg_t = tviz_models.vgg19(pretrained=True).features

        self.down_sample = nn.MaxPool2d(2)

        self.layer_0_1_down = nn.Sequential()
        self.layer_0_1_down.add_module(
            "0",
            nn.Conv2d(
                self.n_image_input_channels, 64, kernel_size=3, stride=1, padding=1
            ),
        )
        for layer in range(1, 4):
            self.layer_0_1_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_2_down = nn.Sequential()
        for layer in range(5, 9):
            self.layer_0_2_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_3_down = nn.Sequential()
        for layer in range(10, 18):
            self.layer_0_3_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_4_down = nn.Sequential()
        for layer in range(19, 27):
            self.layer_0_4_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_5_down = nn.Sequential()
        for layer in range(28, 36):
            self.layer_0_5_down.add_module(str(layer), vgg_t[layer])

        # Head 1
        if self.deconv_decoder:
            # Decoder primarily uses ConvTranspose2d
            self.deconv_0_4 = nn.Sequential()
            self.deconv_0_4.add_module(
                "0",
                nn.ConvTranspose2d(
                    512,
                    256,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_4.add_module("1", nn.ReLU(inplace=True))
            self.deconv_0_4.add_module(
                "2", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            )
            self.deconv_0_4.add_module("3", nn.ReLU(inplace=True))

            self.deconv_0_3 = nn.Sequential()
            self.deconv_0_3.add_module(
                "0",
                nn.ConvTranspose2d(
                    256,
                    128,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_3.add_module("1", nn.ReLU(inplace=True))
            self.deconv_0_3.add_module(
                "2", nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            )
            self.deconv_0_3.add_module("3", nn.ReLU(inplace=True))

            self.deconv_0_2 = nn.Sequential()
            self.deconv_0_2.add_module(
                "0",
                nn.ConvTranspose2d(
                    128,
                    64,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_2.add_module("1", nn.ReLU(inplace=True))
            self.deconv_0_2.add_module(
                "2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            )
            self.deconv_0_2.add_module("3", nn.ReLU(inplace=True))

            self.deconv_0_1 = nn.Sequential()
            self.deconv_0_1.add_module(
                "0",
                nn.ConvTranspose2d(
                    64,
                    64,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_1.add_module("1", nn.ReLU(inplace=True))

        else:
            # Decoder primarily uses Upsampling
#            self.upsample_0_5 = nn.Sequential()
#            self.upsample_0_5.add_module("0", nn.Upsample(scale_factor=2))
#
#            # should this go before the upsample?
#            self.upsample_0_5.add_module(
#                "4", nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
#            )
#            self.upsample_0_5.add_module("5", nn.ReLU(inplace=True))
#            self.upsample_0_5.add_module(
#                "6", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#            )
            
            
            self.upsample_0_4 = nn.Sequential()
            self.upsample_0_4.add_module("0", nn.Upsample(scale_factor=2))

            # should this go before the upsample?
            self.upsample_0_4.add_module(
                "4", nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
            )
            self.upsample_0_4.add_module("5", nn.ReLU(inplace=True))
            self.upsample_0_4.add_module(
                "6", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            )
            
#            self.upsample_0_6 = nn.Sequential()
#            self.upsample_0_6.add_module("0", nn.Upsample(scale_factor=2))
#            self.upsample_0_6.add_module(
#                "4", nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
#            )
#            self.upsample_0_6.add_module("5", nn.ReLU(inplace=True))
#            self.upsample_0_6.add_module(
#                "6", nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
#            )

            self.upsample_0_3 = nn.Sequential()
            self.upsample_0_3.add_module("0", nn.Upsample(scale_factor=2))
            self.upsample_0_3.add_module(
                "4", nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
            )
            self.upsample_0_3.add_module("5", nn.ReLU(inplace=True))
            self.upsample_0_3.add_module(
                "6", nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
            )

            if self.full_output:
                self.upsample_0_2 = nn.Sequential()
                self.upsample_0_2.add_module("0", nn.Upsample(scale_factor=2))
                self.upsample_0_2.add_module(
                    "2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_2.add_module("3", nn.ReLU(inplace=True))
                self.upsample_0_2.add_module(
                    "4", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_2.add_module("5", nn.ReLU(inplace=True))

                self.upsample_0_1 = nn.Sequential()
                self.upsample_0_1.add_module("00", nn.Upsample(scale_factor=2))
                self.upsample_0_1.add_module(
                    "2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_1.add_module("3", nn.ReLU(inplace=True))
                self.upsample_0_1.add_module(
                    "4", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_1.add_module("5", nn.ReLU(inplace=True))

        # Output head - goes from [batch x 64 x height x width] -> [batch x n_keypoints x height x width]
        self.heads_0 = nn.Sequential()
        self.heads_0.add_module(
            "0", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.heads_0.add_module("1", nn.ReLU(inplace=True))
        self.heads_0.add_module(
            "2", nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        )
        self.heads_0.add_module("3", nn.ReLU(inplace=True))
        self.heads_0.add_module(
            "4", nn.Conv2d(32, self.n_keypoints, kernel_size=3, stride=1, padding=1)
        )
        
        self.heads_1 = nn.Sequential()
        self.heads_1.add_module(
            "0", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.heads_1.add_module("1", nn.ReLU(inplace=True))
        self.heads_1.add_module(
            "2", nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        )
        self.heads_1.add_module("3", nn.ReLU(inplace=True))
        self.heads_1.add_module(
            "4", nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        )
        
        self.heads_2 = nn.Sequential()
        self.heads_2.add_module(
            "0", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.heads_2.add_module("1", nn.ReLU(inplace=True))
        self.heads_2.add_module(
            "2", nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        )
        self.heads_2.add_module("3", nn.ReLU(inplace=True))
        self.heads_2.add_module(
            "4", nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        )
        
#        # Spatial softmax output of belief map
#        if self.internalize_spatial_softmax:
#            self.softmax = nn.Sequential()
#            self.softmax.add_module(
#                "0",
#                SoftArgmaxPavlo(
#                    n_keypoints=self.n_keypoints,
#                    learned_beta=self.learned_beta,
#                    initial_beta=self.initial_beta,
#                ),
#            )

    def forward(self, img, pre_img, pre_hm):
#    def forward(self, x):
        
        assert img.shape == pre_img.shape
        assert pre_hm.shape[1] == 1
#        print('img.shape', img.shape)
#        print("pre_img.shape", pre_img.shape)
#        print("pre_hm.shape", pre_hm.shape)
        x = torch.cat((img, pre_img, pre_hm), dim=1)
#        x = img
        # Encoder
        x_0_1 = self.layer_0_1_down(x)
        x_0_1_d = self.down_sample(x_0_1)
        x_0_2 = self.layer_0_2_down(x_0_1_d)
        x_0_2_d = self.down_sample(x_0_2)
        x_0_3 = self.layer_0_3_down(x_0_2_d)
        x_0_3_d = self.down_sample(x_0_3)
        x_0_4 = self.layer_0_4_down(x_0_3_d)
        x_0_4_d = self.down_sample(x_0_4)
        x_0_5 = self.layer_0_5_down(x_0_4_d)

        if self.skip_connections:
            decoder_input = x_0_5 + x_0_4_d
        else:
            decoder_input = x_0_5

        # Decoder
        if self.deconv_decoder:
            y_0_5 = self.deconv_0_4(decoder_input)

            if self.skip_connections:
                y_0_4 = self.deconv_0_3(y_0_5 + x_0_3_d)
            else:
                y_0_4 = self.deconv_0_3(y_0_5)

            if self.skip_connections:
                y_0_3 = self.deconv_0_2(y_0_4 + x_0_2_d)
            else:
                y_0_3 = self.deconv_0_2(y_0_4)

            if self.skip_connections:
                y_0_out = self.deconv_0_1(y_0_3 + x_0_1_d)
            else:
                y_0_out = self.deconv_0_1(y_0_3)

            if self.skip_connections:
                output_head_0 = self.heads_0(y_0_out + x_0_1)
            else:
                output_head_0 = self.heads_0(y_0_out)

        else:
            y_0_5 = self.upsample_0_4(decoder_input)
            # y_0_off = self.upsample_0_5(decoder_input)
             

            if self.skip_connections:
                y_0_out = self.upsample_0_3(y_0_5 + x_0_3_d)
            else:
                y_0_out = self.upsample_0_3(y_0_5)
                # y_0_off = self.upsample_0_6(y_0_off)

            if self.full_output:
                y_0_out = self.upsample_0_2(y_0_out)
                y_0_out = self.upsample_0_1(y_0_out)

            output_head_0 = self.heads_0(y_0_out)
            output_head_1 = self.heads_1(y_0_out)
            output_head_2 = self.heads_2(y_0_out)
            

        # Output heads
        outputs = {}
        outputs["hm"] = output_head_0
        outputs["reg"] = output_head_1
        outputs["tracking"] = output_head_2
        
#        # Spatial softmax output of belief map
#        if self.internalize_spatial_softmax:
#            softmax_output = self.softmax(output_head_0)
#            outputs.append(softmax_output)

        # Return outputs
        return [outputs]




















