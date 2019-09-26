import torch
from detector import *
from torchvision import transforms
import cv2
from PIL import Image
import cfg

if __name__ == '__main__':
    detector = Detector()
    data_for = transforms.Compose([
        transforms.ToTensor()
    ])
    cap = cv2.VideoCapture("D:/ai/ai/train2017/train2017/000000000394.jpg")
    while True:
        # ret 是否读取到图片
        # 图片的数据
        ret, frame = cap.read()
        if ret:
            # OPENCV读出来的数据是BGR格式，这里转换成RGB
            frames = frame[:, :, ::-1]
            image = Image.fromarray(frames, 'RGB')
            width, high = image.size
            x_w = width / 416
            y_h = high / 416
            cropimg = image.resize((416, 416))
            imgdata = data_for(cropimg)
            imgdata = torch.FloatTensor(imgdata).view(-1, 3, 416, 416).cuda()
            y = detector(imgdata, 0.55, cfg.anchors_group)[0]
            print(y)
            for i in y:
                x1 = int((i[0]) * x_w)
                y1 = int((i[1]) * y_h)
                x2 = int((i[2]) * x_w)
                y2 = int((i[3]) * y_h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))
        cv2.imshow('a', frame)
        cv2.waitKey(0)
