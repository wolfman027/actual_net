# 创建训练需要的数据集

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import cfg
import math
import numpy as np

# 做数据归一化，并转换图片大小为：416*416
data_for = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])


# 创建数据集对象
class CocoDataset(Dataset):
    def __init__(self):
        with open(cfg.label_file) as f:
            # 加载每一行数据信息
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}
        # 取出一张图片里的所有信息
        # images/1.jpg 1 12 13 51 18 2 22 31 55 98 2 44 33 62 62
        line = self.dataset[index]
        # 按空格进行切割
        # ['images/1.jpg', '1', '12', '13', '51', '18', '2', '22', '31', '55', '98', '2', '44', '33', '62', '62']、
        strs = line.split()
        # 拼接图片地址+图片名称，获取图片的数据进行加载
        _img_data = Image.open(os.path.join(cfg.img_base_dir, strs[0]))
        # 图片数据转为torch所需要的张量形式，也就是torch可以是别的数据格式
        img_data = data_for(_img_data.convert('RGB'))
        # 获取图片类型和框的数据
        # images/1.jpg 1 12 13 51 18 2 22 31 55 98 2 44 33 62 62 取出 1 12 13 51 18 2 22 31 55 98 2 44 33 62 62
        # [ 1. 12. 13. 51. 18.  2. 22. 31. 55. 98.  2. 44. 33. 62. 62.]
        _boxes = np.array([float(x) for x in strs[1:]])
        # 可以等价这么写
        # _boxes = np.array(list(map(float,strs[1:])))
        # 把类型和相应的框为一组分成列表
        # [array([ 1., 12., 13., 51., 18.]), array([ 2., 22., 31., 55., 98.]), array([ 2., 44., 33., 62., 62.])]
        boxes = np.split(_boxes, len(_boxes)//5)

        # 特征大小，计划框大小
        for feature_size, anchors in cfg.anchors_group.items():
            # 标签初始化
            # (13高，13宽，3（建议框个数），5+10（1个置信度，4个坐标，10分类）)
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 6), dtype=np.float32)
            # print("labels.shape:{}".format(labels[feature_size].shape))
            # 循环解开boxes数据
            for box in boxes:
                cls, cx, cy, w, h = box
                # math.modf 返回小数部分和整数部分
                # 获得中心点的位置
                # 例如中心点(300,200) 300*13/416 = 9.375 200*13/416=6.25
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.img_width)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.img_height)
                # 建议狂有三种
                for i, anchor in enumerate(anchors):
                    # 每个建议框对应的面积
                    # 特征大小为13的第一个建议框的面积为：116*90
                    # 建议框的面积
                    anchor_area = cfg.anchors_group_area[feature_size][i]
                    # 真实宽和高与建议框的比例 真实框/计划狂
                    p_w, p_h = w/anchor[0], h/anchor[1]
                    # 真实数据的面积
                    p_area = w * h
                    # 计算置信度（痛心的IOU（交并））
                    inter = np.minimum(w, anchor[0]) * np.minimum(h,anchor[1])
                    # 交集比并集
                    iou = inter/(p_area+anchor_area-inter)
                    if iou > 0.5:
                        iou = 1
                    else:
                        iou = 0
                    # labels[13][y的索引位置][x的索引位置][第i个框]
                    # HW3,15
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [cx_offset, cy_offset, np.log(p_w), np.log(p_h), iou, int(cls)]
                    )
        # 返回特征图大小
        return labels[13], labels[26], labels[52], img_data


if __name__ == '__main__':
    mydataset = CocoDataset()
    labels_13, labels_26, labels_52, img_data = mydataset.__getitem__(0)
    print(labels_13.shape, labels_26.shape, labels_52.shape, img_data.shape)

