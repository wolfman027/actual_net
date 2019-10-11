# 创建测试数据集
from torch.utils.data import Dataset
import os
import cfg
import torch
from PIL import Image
import numpy as np


class FaceDataset(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        self.dataset = []
        # 打开样本标签文档添加到数据列表中
        self.dataset.extend(open(os.path.join(data_path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(data_path, "part.txt")).readlines())
        self.dataset.extend(open(os.path.join(data_path, "negative.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split()
        # 标签：置信度+偏移量+五官偏移量
        cond = torch.Tensor([int(strs[1])])  # []莫丢，否则指定的是shape
        face_offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
        # 样本：img_data
        img_path = os.path.join(self.path, strs[0])

        img_data = torch.Tensor(np.array(Image.open(img_path)) / 255. - 0.5)
        img_data = img_data.permute(2, 0, 1)  # CWH
        return img_data, cond, face_offset


if __name__ == '__main__':
    dataset = FaceDataset(cfg.img_12_dir)
    print(dataset[0])
