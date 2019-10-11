import torch
import torch.nn as nn
from torch.nn import functional


# 把常用的2个卷积操作简单封装下，原Unet网络在下采样得时候是不加padding的。而我们是加padding的。
class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),    # 添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# 下采样的时候没用池化，直接用卷积代替了池化。
# 如果想要精度提高，那么设置为卷积核是3，padding是1，这样会更好的进行像素融合。
class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        self.downSample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 2, padding=0, stride=2)
        )

    def forward(self, x):
        return self.downSample(x)


# 临近上采样
# 如果想要精度提高，就要改成双线插值
class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()

    def forward(self, x):
        return functional.interpolate(x, scale_factor=2, mode='nearest')



# Unet主网络
class Unet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(Unet, self).__init__()
        # 下采样部分
        self.conv1 = DoubleConv(in_channel, 64)    # 两次卷积，572,572->572,572
        self.pool1 = DownSample(64, 64)   # 卷积代替池化，更好的融合像素，572,572 -> 286,286

        self.conv2 = DoubleConv(64, 128)    # 两次卷积，286,286->286,286
        self.pool2 = DownSample(128, 128)   # 286,286 -> 143,143

        self.conv3 = DoubleConv(128, 256)  # 两次卷积，143,143->143,143
        self.pool3 = DownSample(256, 256)  # 143,143 -> 71,71

        self.conv4 = DoubleConv(256, 512)  # 两次卷积，71,71->71,71
        self.pool4 = DownSample(512, 512)  # 71,71 -> 35,35

        self.conv5 = DoubleConv(512, 1024)  # 两次卷积，35,35->35,35

        # 上采样部分
        # 逆卷积，也可以使用上采样
        self.up6 = UpSample()  # nn.ConvTranspose2d(1024, 512, 2, stride=2) 35,35 -> 70,70
        self.conv6 = DoubleConv(1024+512, 512)      # 70,70 -> 70,70

        self.up7 = UpSample()       # 70,70 -> 140,140
        self.conv7 = DoubleConv(512 + 256, 256)     # 140,140 -> 140,140
        self.up8 = UpSample()     # 140,140 -> 280,280
        self.conv8 = DoubleConv(256 + 128, 128)     # 280,280 -> 280,280
        self.up9 = UpSample()     # 280,280 -> 560,560
        self.conv9 = DoubleConv(128 + 64, 64)     # 560,560 -> 560,560
        self.conv10 = nn.Conv2d(64, out_channel, 1)     # 560,560 -> 560,560

    def forward(self, x):
        # 下采样执行
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        # 上采样执行
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        out = nn.Sigmoid()(c10)
        return out


if __name__ == '__main__':
    unet = Unet()
    x = torch.Tensor(1, 1, 512, 512)
    cond = unet(x)
    print(cond.shape)
    print(cond)


