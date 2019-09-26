# yolo卷积网络

import torch
import torch.nn as nn
from torch.nn import functional


# 上采样，在13*13升到26*26，26*26升到52*52时使用
class UpsampleLayer(nn.Module):
    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return functional.interpolate(x, scale_factor=2, mode='nearest')


# 封装卷积层
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()
        self.sub_model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_model(x)


# 残差层封装
class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()
        self.sub_model = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels//2, 1, 1, 0),
            ConvolutionalLayer(in_channels//2, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.sub_model(x)


# 下采样封装
class DownSamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingLayer, self).__init__()
        self.sub_modle = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_modle(x)


# 最终输出卷积进行整合
class ConvolutionalSet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()
        self.sub_modle = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.sub_modle(x)


# sets 合并
class ConvolutionalSets(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSets, self).__init__()
        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 1, 1),
            ConvolutionalLayer(out_channels, in_channels, 1, 1, 0),
            ConvolutionalLayer(in_channels, out_channels, 3, 1, 1),
            ConvolutionalLayer(out_channels, in_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.sub_module(x)


# 主网络
class MainNet(nn.Module):
    def __init__(self, cls_num):
        super(MainNet, self).__init__()
        self.trunk_52 = nn.Sequential(
            # 2,3,416,416
            ConvolutionalLayer(3, 32, 3, 1, 1),     # 2,32,416,416
            ConvolutionalLayer(32, 64, 3, 2, 1),    # 2,64,208,208

            ResidualLayer(64),  # 2,64,208,208
            DownSamplingLayer(64, 128),  # 2,128,104,104

            ResidualLayer(128),  # 2,128,104,104
            ResidualLayer(128),  # 2,128,104,104
            DownSamplingLayer(128, 256),  # 2,256,52,52

            ResidualLayer(256),  # 2,256,52,52
            ResidualLayer(256),  # 2,256,52,52
            ResidualLayer(256),  # 2,256,52,52
            ResidualLayer(256),  # 2,256,52,52
            ResidualLayer(256),  # 2,256,52,52
            ResidualLayer(256),  # 2,256,52,52
            ResidualLayer(256),  # 2,256,52,52
            ResidualLayer(256)  # 2,256,52,52
        )
        self.trunk_26 = nn.Sequential(
            # 2,256,52,52
            DownSamplingLayer(256, 512),  # 2,512,26,26
            ResidualLayer(512),  # 2,512,26,26
            ResidualLayer(512),  # 2,512,26,26
            ResidualLayer(512),  # 2,512,26,26
            ResidualLayer(512),  # 2,512,26,26
            ResidualLayer(512),  # 2,512,26,26
            ResidualLayer(512),  # 2,512,26,26
            ResidualLayer(512),  # 2,512,26,26
            ResidualLayer(512)  # 2,512,26,26
        )
        self.trunk_13 = nn.Sequential(
            # 2,512,27,40
            DownSamplingLayer(512, 1024),  # 2,1024,13,13
            ResidualLayer(1024),  # 2,1024,13,13
            ResidualLayer(1024),  # 2,1024,13,13
            ResidualLayer(1024),  # 2,1024,13,13
            ResidualLayer(1024)  # 2,1024,13,13
        )
        self.convset_13 = nn.Sequential(
            # 2,1024,13,13
            ConvolutionalSet(1024, 512)  # 2,512,13,13
        )
        self.detetion_13 = nn.Sequential(
            # 2,512,13,13
            ConvolutionalLayer(512, 1024, 3, 1, 1),  # 2,1024,13,13
            nn.Conv2d(1024, 3 * (5 + cls_num), 1, 1, 0)  # 2,3*(5+cls_num),13,13
        )
        self.up_26 = torch.nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),  # 2,256,13,13
            UpsampleLayer()  # 2,256,26,26
        )
        self.convset_26 = nn.Sequential(
            ConvolutionalLayer(768, 256, 1, 1, 0),
            ConvolutionalSets(256, 512)
        )
        self.detetion_26 = torch.nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 3 * (5 + cls_num), 1, 1, 0)
        )
        self.up_52 = torch.nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            UpsampleLayer()
        )
        self.convset_52 = torch.nn.Sequential(
            ConvolutionalLayer(384, 128, 1, 1, 0),
            ConvolutionalSets(128, 256)
        )
        self.detetion_52 = torch.nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 3 * (5 + cls_num), 1, 1, 0)
        )

    def forward(self, x):
        # 2,3,424,640
        h_52 = self.trunk_52(x)  # 2,256,53,80
        h_26 = self.trunk_26(h_52)  # 2,512,27,40
        h_13 = self.trunk_13(h_26)  # 2,1024,14,20

        convset_out_13 = self.convset_13(h_13)  # 2,512,14,20
        detetion_13 = self.detetion_13(convset_out_13)  # 2,15,14,20

        up_out_26 = self.up_26(convset_out_13)  # 2,256,28,40
        route_out_26 = torch.cat((up_out_26, h_26), dim=1)  # #2,512,27,40 + #2,256,28,40 = 2,768,28,40
        convset_out_26 = self.convset_26(route_out_26)  # 2,256,28,40
        detetion_26 = self.detetion_26(convset_out_26)  # 2,15,28,40

        up_out_52 = self.up_52(convset_out_26)  # 2,128,56,80
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)  # 2,128,56,80 + #2,256,53,80 = #2,384,53,80
        convset_out_52 = self.convset_52(route_out_52)  # 2,128,56,80
        detetion_52 = self.detetion_52(convset_out_52)  # 2,15,56,80

        return detetion_13, detetion_26, detetion_52


if __name__ == '__main__':
    trunk = MainNet(80)
    x = torch.Tensor(2, 3, 416, 416)
    y_13, y_26, y_52 = trunk(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)

