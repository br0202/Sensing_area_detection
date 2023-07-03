from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import tclib
from vit.vit_pytorch.vit import ViT


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride,
                                   padding=int(np.floor((self.kernel_size - 1) / 2)))
        self.normalize = nn.BatchNorm2d(num_out_layers)
        self.elu = F.elu

    def forward(self, x):
        x = self.conv_base(x)
        x = self.normalize(x)
        x = self.elu(x)
        # print('x', x.shape)    # x torch.Size([5, 64, 115, 153])
        # return F.elu(x, inplace=True)
        return x

class convblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(convblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return F.max_pool2d((x), self.kernel_size, stride=2, padding=int(np.floor((self.kernel_size-1) / 2)))


class resconv_R(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_R, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_out_layers, 4*num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4*num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4*num_out_layers)
        self.elu = F.elu

    def forward(self, x):
        # do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        x = self.elu(self.normalize(x_out + shortcut))
        return x
        # return F.elu(self.normalize(x_out + shortcut), inplace=True)


class resconv_L(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_L, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_out_layers, 4*num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4*num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4*num_out_layers)
        self.elu = F.elu

    def forward(self, x):
        # do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        x = self.elu(self.normalize(x_out + shortcut))
        return x
        # return F.elu(self.normalize(x_out + shortcut), inplace=True)


class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)
        self.elu = F.elu

    def forward(self, x):
        #         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x
        x = self.elu(self.normalize(x_out + shortcut))
        return x


def resblock_L(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_L(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks - 1):
        layers.append(resconv_L(4 * num_out_layers, num_out_layers, 1))
    layers.append(resconv_L(4 * num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_R(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_R(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks - 1):
        layers.append(resconv_R(4 * num_out_layers, num_out_layers, 1))
    layers.append(resconv_R(4 * num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        return x


class get_segmap(nn.Module):
    def __init__(self, num_in_layers):
        super(get_segmap, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 4, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.normalize = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.normalize(x)   # todo recover
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

class get_pt(nn.Module):
    def __init__(self, num_in_layers):
        super(get_pt, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 4, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.normalize = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.normalize(x)   # todo recover
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class MLP_BNE(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):   # output_dim=4, num_layers=3
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        print('self.layers', self.layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            print('x', x.shape)
        return x


class MLP_OOT(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(200, 400),
        nn.ReLU(),
        nn.Linear(400, 200),
        nn.ReLU(),
        nn.Linear(200, 100))

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)



'''
Mono setting
'''
class mono_vit_mlp(nn.Module):
    def __init__(self, num_in_layers):
        super(mono_vit_mlp, self).__init__()
        # input image vit
        self.v = ViT(
            image_size=896,
            patch_size=32,
            num_classes=512,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        # input points mlp
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512+512, 256), # with points mlp
            # nn.Linear(512, 256),    # without points mlp, just from vit to end results
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points):
        # input image b x 3 x w x h
        # encoder
        img_f = self.v(images)

        # points
        pt_feature = self.layers(points.float())
        fuse = torch.cat((img_f, pt_feature), 1)
        self.left_pred = self.decoder(fuse)

        return self.left_pred

class mono_res_mlp(nn.Module):
    def __init__(self, num_in_layers):
        super(mono_res_mlp, self).__init__()
        # input image b x 6 x w x h
        self.sigmoid = torch.nn.Sigmoid()
        # encoder
        # self.conv1_L = conv(num_in_layers, 3, 3, 2)  # H/2  -   64D
        self.conv1_L = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1_L = maxpool(3)  # H/4  -   64D
        self.conv2_L = resblock_L(64, 64, 3, 2)  # H/8  -  256D
        self.conv3_L = resblock_L(256, 128, 4, 2)  # H/16 -  512D
        self.conv4_L = resblock_L(512, 256, 6, 2)  # H/32 - 1024D
        self.conv5_L = resblock_L(1024, 512, 3, 2)  # H/64 - 2048D

        # decoder
        self.iconv6_L = conv(2048, 512, 3, 1)
        self.iconv5_L = conv(512, 256, 3, 1)
        self.iconv4_L = conv(256, 128, 3, 1)
        self.iconv3_L = conv(128, 64, 3, 1)
        self.iconv2_L = conv(64, 32, 3, 1)
        self.fc1 = nn.Linear(6272, 2128)
        self.fc2 = nn.Linear(2128, 532)

        # LSTM
        # self.rnn = nn.LSTM(100, 512, 4, batch_first=True)
        # self.h0 = torch.randn(2, 3, 20)
        # self.c0 = torch.randn(2, 3, 20)

        # input points  mlp
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(532+512, 256),
            # nn.Linear(532, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points):
        # input image b x 6 x w x h
        # encoder
        x1_L = self.conv1_L(images)
        x_pool1_L = self.pool1_L(x1_L)
        x2_L = self.conv2_L(x_pool1_L)
        x3_L = self.conv3_L(x2_L)    # cross connection
        x4_L = self.conv4_L(x3_L)    # cross connection
        x5_L = self.conv5_L(x4_L)

        iconv6_L = self.iconv6_L(x5_L)
        iconv5_L = self.iconv5_L(iconv6_L)
        iconv4_L = self.iconv4_L(iconv5_L)
        iconv3_L = self.iconv3_L(iconv4_L)
        iconv2_L = self.iconv2_L(iconv3_L)
        f_image = torch.flatten(iconv2_L, 1)

        f_image = self.fc1(f_image)
        f_image = self.fc2(f_image)

        # points
        # pt_feature, (hn, cn) = self.rnn(points.float())
        # print('output', output.shape)    # torch.Size([6, 256])
        pt_feature = self.layers(points.float())
        fuse = torch.cat((f_image, pt_feature), 1)
        self.left_pred = self.decoder(fuse)

        return self.left_pred

class mono_res(nn.Module):
    def __init__(self, num_in_layers):
        super(mono_res, self).__init__()
        # input image b x 3 x w x h
        self.sigmoid = torch.nn.Sigmoid()
        # encoder
        # self.conv1_L = conv(num_in_layers, 3, 3, 2)  # H/2  -   64D
        self.conv1_L = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1_L = maxpool(3)  # H/4  -   64D
        self.conv2_L = resblock_L(64, 64, 3, 2)  # H/8  -  256D
        self.conv3_L = resblock_L(256, 128, 4, 2)  # H/16 -  512D
        self.conv4_L = resblock_L(512, 256, 6, 2)  # H/32 - 1024D
        self.conv5_L = resblock_L(1024, 512, 3, 2)  # H/64 - 2048D

        # decoder
        self.iconv6_L = conv(2048, 512, 3, 1)
        self.iconv5_L = conv(512, 256, 3, 1)
        self.iconv4_L = conv(256, 128, 3, 1)
        self.iconv3_L = conv(128, 64, 3, 1)
        self.iconv2_L = conv(64, 32, 3, 1)
        self.fc1 = nn.Linear(6272, 2128)
        self.fc2 = nn.Linear(2128, 532)

        self.decoder = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(532+512, 256),
            nn.Linear(532, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points):
        # input image b x 6 x w x h
        # encoder
        x1_L = self.conv1_L(images)
        x_pool1_L = self.pool1_L(x1_L)
        x2_L = self.conv2_L(x_pool1_L)
        x3_L = self.conv3_L(x2_L)    # cross connection
        x4_L = self.conv4_L(x3_L)    # cross connection
        x5_L = self.conv5_L(x4_L)

        iconv6_L = self.iconv6_L(x5_L)
        iconv5_L = self.iconv5_L(iconv6_L)
        iconv4_L = self.iconv4_L(iconv5_L)
        iconv3_L = self.iconv3_L(iconv4_L)
        iconv2_L = self.iconv2_L(iconv3_L)
        f_image = torch.flatten(iconv2_L, 1)

        f_image = self.fc1(f_image)
        f_image = self.fc2(f_image)

        # points
        # pt_feature, (hn, cn) = self.rnn(points.float())
        # print('output', output.shape)    # torch.Size([6, 256])
        # pt_feature = self.layers(points.float())
        # fuse = torch.cat((f_image, pt_feature), 1)
        self.left_pred = self.decoder(f_image)

        return self.left_pred

class mono_vit(nn.Module):
    def __init__(self, num_in_layers):
        super(mono_vit, self).__init__()
        # input image b x 3 x w x h
        self.v = ViT(
            image_size=896,
            patch_size=32,
            num_classes=512,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(512+512, 256), # with points mlp
            nn.Linear(512, 256),  # without points mlp, just from vit to end results
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points):
        # input image b x 3 x w x h
        # encoder
        img_f = self.v(images)

        # points
        # pt_feature, (hn, cn) = self.rnn(points.float())
        # pt_feature = self.layers(points.float())
        # fuse = torch.cat((img_f, pt_feature), 1)
        # self.left_pred = self.decoder(fuse)
        self.left_pred = self.decoder(img_f)

        return self.left_pred

'''
Stereo setting
'''
class stereo_vit_mlp(nn.Module):
    def __init__(self, num_in_layers):
        super(stereo_vit_mlp, self).__init__()
        # input image b x 6 x w x h
        self.v = ViT(
            image_size=896,
            patch_size=32,
            num_classes=512,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        # encoder
        self.conv1_L = conv(num_in_layers, 3, 3, 2)  # H/2  -   64D

        # input points
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512+512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points):
        # input image b x 6 x w x h
        # encoder
        img_f = self.conv1_L(images)
        img_f = self.v(img_f)
        # points
        pt_feature = self.layers(points.float())
        fuse = torch.cat((img_f, pt_feature), 1)
        self.left_pred = self.decoder(fuse)

        return self.left_pred

class stereo_res_mlp(nn.Module):
    def __init__(self, num_in_layers):
        super(stereo_res_mlp, self).__init__()
        # input image b x 6 x w x h
        self.sigmoid = torch.nn.Sigmoid()
        # encoder
        # self.conv1_L = conv(num_in_layers, 3, 3, 2)  # H/2  -   64D
        self.conv1_L = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1_L = maxpool(3)  # H/4  -   64D
        self.conv2_L = resblock_L(64, 64, 3, 2)  # H/8  -  256D
        self.conv3_L = resblock_L(256, 128, 4, 2)  # H/16 -  512D
        self.conv4_L = resblock_L(512, 256, 6, 2)  # H/32 - 1024D
        self.conv5_L = resblock_L(1024, 512, 3, 2)  # H/64 - 2048D

        # decoder
        self.iconv6_L = conv(2048, 512, 3, 1)
        self.iconv5_L = conv(512, 256, 3, 1)
        self.iconv4_L = conv(256, 128, 3, 1)
        self.iconv3_L = conv(128, 64, 3, 1)
        self.iconv2_L = conv(64, 32, 3, 1)
        self.fc1 = nn.Linear(6272, 2128)
        self.fc2 = nn.Linear(2128, 532)

        # LSTM
        # self.rnn = nn.LSTM(100, 512, 4, batch_first=True)
        # self.h0 = torch.randn(2, 3, 20)
        # self.c0 = torch.randn(2, 3, 20)

        # input points
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(532+512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points):
        # input image b x 6 x w x h
        # encoder
        x1_L = self.conv1_L(images)
        x_pool1_L = self.pool1_L(x1_L)
        x2_L = self.conv2_L(x_pool1_L)
        x3_L = self.conv3_L(x2_L)    # cross connection
        x4_L = self.conv4_L(x3_L)    # cross connection
        x5_L = self.conv5_L(x4_L)

        iconv6_L = self.iconv6_L(x5_L)
        iconv5_L = self.iconv5_L(iconv6_L)
        iconv4_L = self.iconv4_L(iconv5_L)
        iconv3_L = self.iconv3_L(iconv4_L)
        iconv2_L = self.iconv2_L(iconv3_L)
        f_image = torch.flatten(iconv2_L, 1)
        f_image = self.fc1(f_image)
        f_image = self.fc2(f_image)

        # points
        pt_feature = self.layers(points.float())
        fuse = torch.cat((f_image, pt_feature), 1)
        self.left_pred = self.decoder(fuse)

        return self.left_pred

class stereo_res(nn.Module):
    def __init__(self, num_in_layers):
        super(stereo_res, self).__init__()
        # input image b x 6 x w x h
        self.sigmoid = torch.nn.Sigmoid()
        # encoder
        # self.conv1_L = conv(num_in_layers, 3, 3, 2)  # H/2  -   64D
        self.conv1_L = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1_L = maxpool(3)  # H/4  -   64D
        self.conv2_L = resblock_L(64, 64, 3, 2)  # H/8  -  256D
        self.conv3_L = resblock_L(256, 128, 4, 2)  # H/16 -  512D
        self.conv4_L = resblock_L(512, 256, 6, 2)  # H/32 - 1024D
        self.conv5_L = resblock_L(1024, 512, 3, 2)  # H/64 - 2048D

        # decoder
        self.iconv6_L = conv(2048, 512, 3, 1)
        self.iconv5_L = conv(512, 256, 3, 1)
        self.iconv4_L = conv(256, 128, 3, 1)
        self.iconv3_L = conv(128, 64, 3, 1)
        self.iconv2_L = conv(64, 32, 3, 1)
        self.fc1 = nn.Linear(6272, 2128)
        self.fc2 = nn.Linear(2128, 532)


        self.decoder = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(532+512, 256),
            nn.Linear(532, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points):
        # input image b x 6 x w x h
        # encoder
        x1_L = self.conv1_L(images)
        x_pool1_L = self.pool1_L(x1_L)
        x2_L = self.conv2_L(x_pool1_L)
        x3_L = self.conv3_L(x2_L)    # cross connection
        x4_L = self.conv4_L(x3_L)    # cross connection
        x5_L = self.conv5_L(x4_L)

        iconv6_L = self.iconv6_L(x5_L)
        iconv5_L = self.iconv5_L(iconv6_L)
        iconv4_L = self.iconv4_L(iconv5_L)
        iconv3_L = self.iconv3_L(iconv4_L)
        iconv2_L = self.iconv2_L(iconv3_L)
        f_image = torch.flatten(iconv2_L, 1)

        f_image = self.fc1(f_image)
        f_image = self.fc2(f_image)

        # points
        # pt_feature, (hn, cn) = self.rnn(points.float())
        # print('output', output.shape)    # torch.Size([6, 256])
        # pt_feature = self.layers(points.float())
        # fuse = torch.cat((f_image, pt_feature), 1)
        self.left_pred = self.decoder(f_image)

        return self.left_pred

class stereo_vit(nn.Module):
    def __init__(self, num_in_layers):
        super(stereo_vit, self).__init__()
        # input image b x 6 x w x h
        self.v = ViT(
            image_size=896,
            patch_size=32,
            num_classes=512,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        # encoder
        self.conv1_L = conv(num_in_layers, 3, 3, 2)  # H/2  -   64D

        self.decoder = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(512+512, 256), # with points mlp
            nn.Linear(512, 256),    # without points mlp, just from vit to end results
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points):
        # input image b x 6 x w x h
        # encoder
        img_f = self.conv1_L(images)
        img_f = self.v(img_f)

        # points
        # pt_feature, (hn, cn) = self.rnn(points.float())
        # pt_feature = self.layers(points.float())
        # fuse = torch.cat((img_f, pt_feature), 1)
        # self.left_pred = self.decoder(fuse)
        self.left_pred = self.decoder(img_f)


        return self.left_pred


# stereo + lstm
class stereo_res_lstm(nn.Module):
    def __init__(self, num_in_layers):
        super(stereo_res_lstm, self).__init__()
        # input image b x 6 x w x h
        self.sigmoid = torch.nn.Sigmoid()
        # encoder
        # self.conv1_L = conv(num_in_layers, 3, 3, 2)  # H/2  -   64D
        self.conv1_L = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1_L = maxpool(3)  # H/4  -   64D
        self.conv2_L = resblock_L(64, 64, 3, 2)  # H/8  -  256D
        self.conv3_L = resblock_L(256, 128, 4, 2)  # H/16 -  512D
        self.conv4_L = resblock_L(512, 256, 6, 2)  # H/32 - 1024D
        self.conv5_L = resblock_L(1024, 512, 3, 2)  # H/64 - 2048D

        # decoder
        self.iconv6_L = conv(2048, 512, 3, 1)
        self.iconv5_L = conv(512, 256, 3, 1)
        self.iconv4_L = conv(256, 128, 3, 1)
        self.iconv3_L = conv(128, 64, 3, 1)
        self.iconv2_L = conv(64, 32, 3, 1)
        self.fc1 = nn.Linear(6272, 2128)
        self.fc2 = nn.Linear(2128, 532)

        # LSTM (input points)
        self.rnn = nn.LSTM(100, 512, 4, batch_first=True)
        # self.h0 = torch.randn(2, 3, 20)
        # self.c0 = torch.randn(2, 3, 20)

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(532+512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points):
        # input image b x 6 x w x h
        # encoder
        x1_L = self.conv1_L(images)
        x_pool1_L = self.pool1_L(x1_L)
        x2_L = self.conv2_L(x_pool1_L)
        x3_L = self.conv3_L(x2_L)    # cross connection
        x4_L = self.conv4_L(x3_L)    # cross connection
        x5_L = self.conv5_L(x4_L)

        iconv6_L = self.iconv6_L(x5_L)
        iconv5_L = self.iconv5_L(iconv6_L)
        iconv4_L = self.iconv4_L(iconv5_L)
        iconv3_L = self.iconv3_L(iconv4_L)
        iconv2_L = self.iconv2_L(iconv3_L)
        f_image = torch.flatten(iconv2_L, 1)

        f_image = self.fc1(f_image)
        f_image = self.fc2(f_image)

        # points
        pt_feature, (hn, cn) = self.rnn(points.float())
        fuse = torch.cat((f_image, pt_feature), 1)
        self.left_pred = self.decoder(fuse)

        return self.left_pred


class stereo_vit_lstm(nn.Module):
    def __init__(self, num_in_layers):
        super(stereo_vit_lstm, self).__init__()
        # input image b x 6 x w x h
        self.v = ViT(
            image_size=896,
            patch_size=32,
            num_classes=512,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        # encoder
        self.conv1_L = conv(num_in_layers, 3, 3, 2)  # H/2  -   64D

        # LSTM (input points)
        self.rnn = nn.LSTM(100, 512, 4, batch_first=True)
        # self.h0 = torch.randn(2, 3, 20)
        # self.c0 = torch.randn(2, 3, 20)

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512+512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, images, points):
        # input image b x 6 x w x h
        # encoder
        img_f = self.conv1_L(images)
        img_f = self.v(img_f)

        # points
        pt_feature, (hn, cn) = self.rnn(points.float())
        fuse = torch.cat((img_f, pt_feature), 1)
        self.left_pred = self.decoder(fuse)

        return self.left_pred


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)

