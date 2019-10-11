# MTCNN的使用
# 流程：图像->缩放->P网络（NMS和边界框回归->R网路(NMS和边界框回归)->O网络(NMS和边界框回归)
import nets
import torch
from torchvision import transforms
import time
import cfg
import os
from PIL import Image
import utils
import numpy as np
from PIL import ImageDraw


# 网络调参
# P网络
p_cls = 0.6
p_nms = 0.5

# R网络
r_cls = 0.6
r_nms = 0.5

# O网络
o_cls = 0.3
o_nms = 0.5


# 侦测器
class Detector:
    # 初始化时加载三个网络的权重(训练好的)，cuda默认设为True
    def __init__(self, pnet_param, rnet_param, onet_param, isCuda=True):
        self.isCuda = isCuda
        # 初始化网络
        self.pnet = nets.P_Net()
        self.rnet = nets.R_Net()
        self.onet = nets.O_Net()
        # 网络加载到cuda
        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        # 网络加载保存好的参数
        self.pnet.load_state_dict(torch.load(pnet_param))
        # self.rnet.load_state_dict(torch.load(rnet_param))
        # self.onet.load_state_dict(torch.load(onet_param))
        # 训练网络里有BN（批归一化时），要调用eval方法，使用是不用BN，dropout方法
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.data_for = transforms.Compose([
            transforms.ToTensor()
        ])

    def detector(self, image):
        # 监测图片
        # P网络监测
        start_time = time.time()
        pnet_boxes = self.__pnet_detector(image)
        if pnet_boxes.shape[0] == 0:           # 若P网络没有人脸时，避免数据出错，返回一个新数组
            return np.array([])
        end_time = time.time()                 # 计时结束
        t_pnet = end_time - start_time         # P网络所占用的时间差
        print(t_pnet)
        return pnet_boxes                    # p网络检测出的框

        # R网络检测-------2nd
        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)  # 传入原图，P网络的一些框，根据这些框在原图上抠图
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time
        print(t_rnet)
        # return rnet_boxes

        # O网络检测--------3rd
        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)  # 把原图和R网络里的框传到O网络里去
        if onet_boxes.shape[0] == 0:  # 若P网络没有人脸时，避免数据出错，返回一个新数组
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        # 三网络检测的总时间
        t_sum = t_pnet + t_rnet + t_onet
        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))
        return onet_boxes

    def __pnet_detector(self, image):
        # P网络全部是卷积，与输入图片大小无关，可输出任意形状图片
        boxes = []
        w, h = image.size
        min_side_len = min(w, h)     # 获取图片最小边长
        scale = 1   # 初始缩放比例（为1时不缩放）:得到不同分辨率的图片
        while min_side_len > 12:    # 直到缩放到小于等于12时停止
            img_data = self.data_for(image.convert('RGB'))     # 将图片数组转为张量
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)  # 在“批次”上升维（测试时传的不止一张图片） NCHW
            # 通过网络得到：置信度，人脸框坐标，五官点
            _cond, _offset_face, _offset_facial = self.pnet(img_data)  # 返回多个置信度和偏移量
            # [245, 203]：特征图的尺寸：H,W
            cond = _cond[0][0].cpu().data
            # [4, 245, 203]：人脸坐标特征图的通道、尺寸:C,H,W
            offset_face = _offset_face[0].cpu().data
            # [10, 245, 203]：五官坐标特征图的通道、尺寸:C,H,W
            offset_facial = _offset_facial[0].cpu().data
            # 置信度大于0.6的框索引；把P网络输出，看有没没框到的人脸，若没框到人脸，说明网络没训练好；或者置信度给高了、调低
            # idxs：tensor([[ 19,  95],[ 20,  95]])
            idxs = torch.nonzero(torch.gt(cond, p_cls))
            boxes_ = self.__box(idxs, cond, offset_face, offset_facial, scale)
            if len(boxes) > 0:
                boxes = torch.cat((boxes, boxes_), dim=0)
            else:
                boxes = boxes_
            scale *= 0.7  # 缩放图片:循环控制条件
            _w = int(w * scale)  # 新的宽度
            _h = int(h * scale)
            image = image.resize((_w, _h))  # 根据缩放后的宽和高，对图片进行缩放
            min_side_len = min(_w, _h)  # 重新获取最小宽高
        return utils.nms(np.array(boxes), p_nms)  # 返回框框，原阈值给p_nms=0.5（iou为0.5），尽可能保留IOU小于0.5的一些框下来，若网络训练的好，值可以给低些

    # 特征反算：将回归量还原到原图上去，根据特征图反算的到原图建议框
    # p网络池化步长为2
    def __box(self, idxs, cond, offset_face, offset_facial, scale, stride=2, side_len=12):
        _x1 = (idxs[:, 1].float() * stride) / scale  # 索引乘以步长，除以缩放比例；★特征反算时“行索引，索引互换”，原为[0]
        _y1 = (idxs[:, 0].float() * stride) / scale
        _x2 = (idxs[:, 1].float() * stride + side_len) / scale
        _y2 = (idxs[:, 0].float() * stride + side_len) / scale
        # 人脸所在区域建议框的宽和高
        ow = _x2 - _x1
        oh = _y2 - _y1
        # 根据idxs行索引与列索引，找到对应偏移量△δ:[x1,y1,x2,y2]
        _offset_face = offset_face[:, idxs[:, 0], idxs[:, 1]]
        x1 = _x1 + ow * _offset_face[0]  # 根据偏移量算实际框的位置，x1=x1_+w*△δ；生样时为:△δ=x1-x1_/w
        y1 = _y1 + oh * _offset_face[1]
        x2 = _x2 + ow * _offset_face[2]
        y2 = _y2 + oh * _offset_face[3]

        # 根据idxs行索引与列索引，找到对应偏移量△δ:[x1,y1,x2,y2]
        _offset_facial = offset_facial[:, idxs[:, 0], idxs[:, 1]]
        efteye_x = _x1 + ow * _offset_facial[0]      # 左眼坐标点x
        lefteye_y = _y1 + oh * _offset_facial[1]      # 左眼坐标点y
        righteye_x = _x1 + ow * _offset_facial[2]      # 右眼坐标点x
        righteye_y = _y1 + oh * _offset_facial[3]      # 右眼坐标点y
        nose_x = _x1 + ow * _offset_facial[4]      # 鼻子坐标点x
        nose_y = _y1 + oh * _offset_facial[5]      # 鼻子坐标点y
        leftmouth_x = _x1 + ow * _offset_facial[6]      # 左嘴角坐标点x
        leftmouth_y = _y1 + oh * _offset_facial[7]      # 左嘴角坐标点y
        rightmouth_x = _x1 + ow * _offset_facial[8]      # 右嘴角坐标点x
        rightmouth_y = _y1 + oh * _offset_facial[9]      # 右嘴角坐标点y
        cls = cond[idxs[:, 0], idxs[:, 1]]
        boxes_ = torch.cat((x1.unsqueeze_(0),
                            y1.unsqueeze_(0),
                            x2.unsqueeze_(0),
                            y2.unsqueeze_(0),
                            cls.unsqueeze_(0),
                            efteye_x.unsqueeze_(0),
                            lefteye_y.unsqueeze_(0),
                            righteye_x.unsqueeze_(0),
                            righteye_y.unsqueeze_(0),
                            nose_x.unsqueeze_(0),
                            nose_y.unsqueeze_(0),
                            leftmouth_x.unsqueeze_(0),
                            leftmouth_y.unsqueeze_(0),
                            rightmouth_x.unsqueeze_(0),
                            rightmouth_y.unsqueeze_(0)
                            ), dim=0)
        boxes = boxes_.permute(1, 0)
        return boxes

    def __rnet_detect(self, image, pnet_boxes):
        # 创建空列表，存放抠图
        _img_dataset = []
        # 给p网络输出的框，找出中心点，沿着最大边长的两边扩充成“正方形”，再抠图
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        # 遍历每个框，每个框返回框4个坐标点，抠图，放缩，数据类型转换，添加列表
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            img = image.crop((_x1, _y1, _x2, _y2))      # 根据4个坐标点抠图
            img = img.resize((24, 24))      # 放缩在固尺寸
            img_data = self.data_for(img)       # 将图片数组转成张量
            _img_dataset.append(img_data)
        # stack堆叠(默认在0轴)，此处相当数据类型转换，list chw —> nchw
        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()
        # 通过r网络训练
        _cond, _offset_face, _offset_facial = self.rnet(img_dataset)
        cond = _cond.cpu().data.numpy()     # 将gpu上的数据放到cpu上去，在转成numpy数组
        offset_face = _offset_face.cpu().data.numpy()
        offset_facial = _offset_facial.cpu().data.numpy()
        boxes = []
        # 原置信度0.6是偏低的，时候很多框并没有用(可打印出来观察)，
        # 可以适当调高些；idxs置信度框大于0.6的索引；
        # ★返回idxs:0轴上索引[0,1]，_:1轴上索引[0,0]，共同决定元素位置，见例子3
        idxs, _ = np.where(cond > r_cls)
        # # 得到框的数据
        # _box = _pnet_boxes[idxs]
        # # 框的两个坐标点
        # _x1 = _box[:, 0].astype(np.int32)
        # _y1 = _box[:, 1].astype(np.int32)
        # _x2 = _box[:, 2].astype(np.int32)
        # _y2 = _box[:, 3].astype(np.int32)
        # # 框的宽和高
        # ow = _x2 - _x1
        # oh = _y2 - _y1
        # # 实际框的坐标点
        # x1 = _x1 + ow * offset_face[idxs][:, 0]  # 实际框的坐标点
        # y1 = _y1 + oh * offset_face[idxs][:, 1]
        # x2 = _x2 + ow * offset_face[idxs][:, 2]
        # y2 = _y2 + oh * offset_face[idxs][:, 3]
        # cls = cond[idxs][:, 0]
        # boxes_ = np.array([x1, y1, x2, y2, cls], dtype=np.float64)
        # boxes_aaaaaaa = np.swapaxes(boxes_, 1, 0)


        print(idxs, _)
        for idx in idxs:    # 根据索引，遍历符合条件的框；1轴上的索引，恰为符合条件的置信度索引（0轴上索引此处用不到）
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1 # 基准框的宽
            oh = _y2 - _y1

            x1 = _x1 + ow * offset_face[idx][0]     # 实际框的坐标点
            y1 = _y1 + oh * offset_face[idx][1]
            x2 = _x2 + ow * offset_face[idx][2]
            y2 = _y2 + oh * offset_face[idx][3]

            # efteye_x = _x1 + offset_facial[0]  # 左眼坐标点x
            # lefteye_y = _y1 + offset_facial[1]  # 左眼坐标点y
            # righteye_x = _x1 + offset_facial[2]  # 右眼坐标点x
            # righteye_y = _y1 + offset_facial[3]  # 右眼坐标点y
            # nose_x = _x1 + offset_facial[4]  # 鼻子坐标点x
            # nose_y = _y1 + offset_facial[5]  # 鼻子坐标点y
            # leftmouth_x = _x1 + offset_facial[6]  # 左嘴角坐标点x
            # leftmouth_y = _y1 + offset_facial[7]  # 左嘴角坐标点y
            # rightmouth_x = _x1 + offset_facial[8]  # 右嘴角坐标点x
            # rightmouth_y = _y1 + offset_facial[9]  # 右嘴角坐标点y

            boxes.append([x1, y1, x2, y2, cond[idx][0]
                          ])    # 返回4个坐标点和置信度
        return utils.nms(np.array(boxes), r_nms)    # 原r_nms为0.5（0.5要往小调），上面的0.6要往大调;小于0.5的框被保留下来

    # 创建O网络检测函数
    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = [] # 创建列表，存放抠图r
        _rnet_boxes = utils.convert_to_square(rnet_boxes) # 给r网络输出的框，找出中心点，沿着最大边长的两边扩充成“正方形”
        for _box in _rnet_boxes: # 遍历R网络筛选出来的框，计算坐标，抠图，缩放，数据类型转换，添加列表，堆叠
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))  # 根据坐标点“抠图”
            img = img.resize((48, 48))
            img_data = self.data_for(img)   # 将抠出的图转成张量
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset) # 堆叠，此处相当数据格式转换，见例子2
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset, _offset_facial = self.onet(img_dataset)
        cls = _cls.cpu().data.numpy()       # (1, 1)
        offset = _offset.cpu().data.numpy() # (1, 4)

        boxes = [] # 存放o网络的计算结果
        idxs, _ = np.where(cls > o_cls) # 原o_cls为0.97是偏低的，最后要达到标准置信度要达到0.99999，这里可以写成0.99998，这样的话出来就全是人脸;留下置信度大于0.97的框；★返回idxs:0轴上索引[0]，_:1轴上索引[0]，共同决定元素位置，见例子3
        for idx in idxs: # 根据索引，遍历符合条件的框；1轴上的索引，恰为符合条件的置信度索引（0轴上索引此处用不到）
            _box = _rnet_boxes[idx] # 以R网络做为基准框
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1 # 框的基准宽，框是“方”的，ow=oh
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0] # O网络最终生成的框的坐标；生样，偏移量△δ=x1-_x1/w*side_len
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]]) #返回4个坐标点和1个置信度

        return utils.nms(np.array(boxes), o_nms, isMin=True) # 用最小面积的IOU；原o_nms(IOU)为小于0.7的框被保留下来


if __name__ == '__main__':
    img_path = cfg.test_img
    for i in os.listdir(img_path):
        detector = Detector(cfg.save_12_params_dir, cfg.save_24_params_dir, cfg.save_48_params_dir)
        with Image.open(os.path.join(img_path, i)) as im:    # 打开图片
            boxes = detector.detector(im)
            print("size:", im.size)
            imDraw = ImageDraw.Draw(im)
            for box in boxes: # 多个框，没循环一次框一个人脸
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                print((x1, y1, x2, y2))
                print("conf:", box[4]) # 置信度
                imDraw.rectangle((x1, y1, x2, y2), outline='red')
                #im.show() # 每循环一次框一个人脸
            im.show()
            # exit()













