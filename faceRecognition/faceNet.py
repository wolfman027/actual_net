import torchvision.models as models
import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np


# arc softmax
class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super(Arcsoftmax, self).__init__()
        # 设置w的初始值
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)))
        self.func = nn.Softmax()

    def forward(self, x, s=1, m=0.2):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)
        # 防止梯度爆炸
        # cosa = 得到两个向量的cos角度
        # xw = ||x||2 * ||w||2 * cosa
        cosa = torch.matmul(x_norm, w_norm) / 10
        # 得到角度的值
        a = torch.acos(cosa)
        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(
            s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))
        return arcsoftmax


# 创建人脸网络
class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        # 加载restnet50网络
        self.sub_net = nn.Sequential(
            models.resnet50(pretrained=True)
        )
        # 特征网络
        self.feature_net = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, 512, bias=False)
        )
        self.arc_softmax = Arcsoftmax(512, 8)

    def forward(self, x):
        y = self.sub_net(x)
        feature = self.feature_net(y)
        return feature, self.arc_softmax(feature, 1, 1)

    def encode(self, x):
        return self.sub_net(x)


def compare(face1, face2):
    dot = np.sum(np.multiply(face1.cpu().data.numpy(), face2.cpu().data.numpy()), axis=1)
    # np.linalg.norm 求得范数
    norm = np.linalg.norm(face1.cpu().data.numpy(), axis=1) * np.linalg.norm(face2.cpu().data.numpy(), axis=1)
    dist = dot / norm
    return dist


    # face1_norm = F.normalize(face1, dim=1)
    # face2_norm = F.normalize(face2, dim=1)
    # cosa = torch.dot(face1_norm, face2_norm)
    # return cosa     # 如果是负数得话，直接归0就行了。这个值就是两个人相似得程度












