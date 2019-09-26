# 侦测网络
from darknet53 import *
import utils
import cfg
import torch

device = torch.device(utils.getDevice())


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.net = MainNet(cfg.class_num).to(device)
        a = torch.load('data/ckpt-225.pt')
        self.net.load_state_dict(a)
        self.net.eval()

    def forward(self, input, thresh, anchors):
        # thresh 计算置信度的时候要达到的阈值
        # 通过网络得到输出NCHW
        output_13, output_26, output_52 = self.net(input.to(device))
        # 通过过滤方法，得到置信度大于阈值的位置
        # 得到置信度大于阈值的位置-idxs_13：大于1的数量，位置，例如：[[0,6,4,2],[0, 6, 5, 2]]，shape：[12,4]
        # 位置上的值：大于1的数量，5+cls。shape：[12,85]
        idxs_13, vecs_13 = self._filter(output_13, thresh)
        # 得到 x1, y1, x2, y2, c 置信度, cls 类别, n 那个照片
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        boxes_all = torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

        # 同一张图片得不同分类分开坐NMS
        last_boxes = []
        for n in range(input.size(0)):
            n_boxes = []
            boxes_n = boxes_all[boxes_all[:, 6] == n]
            print(boxes_n)
            for cls in range(cfg.class_num):
                boxes_c = boxes_n[boxes_n[:, 5] == cls]
                if boxes_c.size(0) > 0:
                    n_boxes.extend(utils.nms(boxes_c, 0.3))
                else:
                    pass
            last_boxes.append(torch.stack(n_boxes))

        return last_boxes

    def _filter(self, output, thresh):
        # NCHW - > NHWC
        output = output.permute(0, 2, 3, 1)
        # NHWC -> NHW3*
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        # 数据放回cpu中
        output = output.cpu()
        # 数据转换
        torch.sigmoid_(output[..., 4])      # 置信度加sigmoid函数
        torch.sigmoid_(output[..., 0:2])   # 中心点加sigmoid激活
        # 计算置信度大于阈值的判断并返回数组 （NHW3）
        mask = output[..., 4] > thresh
        # 得到不为0数据的坐标NHWC
        idxs = mask.nonzero()   #得到大于阈值不为零的掩码，也就是位置
        vecs = output[mask]     #得到大于阈值的数据
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        if idxs.size(0) == 0:
            return torch.Tensor([])
        anchors = torch.Tensor(anchors)
        n = idxs[:, 0]  # 所属的图片
        a = idxs[:, 3]  # 建议框
        c = vecs[:, 4]  # 置信度
        cls = torch.argmax(vecs[:, 5:], dim=1)  # 类别
        cy = (idxs[:, 1].float() + vecs[:, 1]) * t  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 0]) * t  # 原图的中心点x
        w = anchors[a, 0] * torch.exp(vecs[:, 2])   # 宽
        h = anchors[a, 1] * torch.exp(vecs[:, 3])   # 高
        w0_5, h0_5 = w / 2, h / 2
        x1, y1, x2, y2 = cx - w0_5, cy - h0_5, cx + w0_5, cy + h0_5
        return torch.stack([x1, y1, x2, y2, c, cls.float(), n.float()], dim=1)

    def export(self):
        # 到处如果报错。需要将pytorch1.1.0版本降到1.0.1版本
        # 导出网络未onnx格式给别人使用
        dummy_input = torch.FloatTensor(1, 3, 416, 416).cuda()
        torch.onnx.export(self.net, dummy_input, "darknet53.onnx", verbose=True)


if __name__ == '__main__':
    detector = Detector()
    detector.export()






