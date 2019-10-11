# MTCNN三个神经网络
import torch
import torch.nn as nn


# P网络
class P_Net(nn.Module):
    def __init__(self):
        super(P_Net, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),  # n,10,12,12
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # n,10,5,5
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1),    # n,16,3,3
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),    # n,32,1,1
            nn.PReLU()
        )
        self.conv4_1 = nn.Conv2d(32, 1, 1, 1)   # n,1,1,1
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)   # n,4,1,1
        self.conv4_3 = nn.Conv2d(32, 10, 1, 1)   # n,10,1,1

    def forward(self, x):
        x = self.pre_conv(x)
        cond = torch.sigmoid(self.conv4_1(x))  # 置信度用sigmoid激活(用BCEloos时先要用sigmoid激活)
        offset_face = self.conv4_2(x)
        offset_facial = self.conv4_3(x)
        return cond, offset_face, offset_facial


# R网络
class R_Net(nn.Module):
    def __init__(self):
        super(R_Net, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1, padding=1),  # n,28,24,24
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # n,28,11,11
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1),    # n,48,9,9
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # n,48,4,4
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1),    # n,64,3,3
            nn.PReLU()
        )
        self.full = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),     # n,128
            nn.PReLU()
        )
        self.full_conf = nn.Linear(128, 1)
        self.full_offset_face = nn.Linear(128, 4)
        self.full_offset_facial = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pre_conv(x)
        x = x.view(x.size(0), -1)
        x = self.full(x)
        cond = torch.sigmoid(self.full_conf(x))  # 置信度用sigmoid激活(用BCEloos时先要用sigmoid激活)
        offset_face = self.full_offset_face(x)
        offset_facial = self.full_offset_facial(x)
        return cond, offset_face, offset_facial


# O网络
class O_Net(nn.Module):
    def __init__(self):
        super(O_Net, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),  # n,32,48,48
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # n,32,23,23
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),    # n,64,21,21
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # n,64,10,10
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),    # n,64,8,8
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # n,64,4,4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1)     # n,128,3,3
        )
        self.full = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),     # n,256
            nn.PReLU()
        )
        self.full_conf = nn.Linear(256, 1)
        self.full_offset_face = nn.Linear(256, 4)
        self.full_offset_facial = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pre_conv(x)
        x = x.view(x.size(0), -1)
        x = self.full(x)
        cond = torch.sigmoid(self.full_conf(x))  # 置信度用sigmoid激活(用BCEloos时先要用sigmoid激活)
        offset_face = self.full_offset_face(x)
        offset_facial = self.full_offset_facial(x)
        return cond, offset_face, offset_facial


if __name__ == '__main__':
    pnet = P_Net()
    x = torch.Tensor(2, 3, 12, 12)
    cond, offset_face, offset_facial = pnet(x)
    print(cond.shape)
    print(offset_face.shape)
    print(offset_facial.shape)
    rnet = R_Net()
    x = torch.Tensor(2, 3, 24, 24)
    cond, offset_face, offset_facial = rnet(x)
    print(cond.shape)
    print(offset_face.shape)
    print(offset_facial.shape)
    onet = O_Net()
    x = torch.Tensor(2, 3, 48, 48)
    cond, offset_face, offset_facial = onet(x)
    print(cond.shape)
    print(offset_face.shape)
    print(offset_facial.shape)








