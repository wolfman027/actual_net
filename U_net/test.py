import torch
import numpy as np
from Unet import Unet
import cv2
from PIL import Image
import cfg
from matplotlib import image

if __name__ == '__main__':
    net = Unet(1, 1).cuda()
    net.load_state_dict(torch.load(cfg.train_data_params_pt_src))
    # 读取测试图片
    x = image.imread(r'E:\net_data\unet\trainData\data\7.png')
    img = x
    x = torch.Tensor(x)
    x = x.view(-1, 1, 512, 512)
    x = x.cuda()
    # 通过网络得到掩码
    out = net(x)    # 1,1,512,512
    # 转换：[1, 1, 512, 512] -> [512, 512]
    out = torch.reshape(out.cpu(), [512, 512])
    # detach的方法，将variable参数从网络中隔离开，不参与参数更新
    out = out.cpu().detach().numpy()
    # 转换，确保为[512, 512]
    test_img = np.reshape(out, [512, 512])
    # 得到掩码值：[512, 512] 里边值为 True 或 False
    mask_obj = test_img > 0.85
    # 得到掩码值：[512, 512] 里边值为 True 或 False
    mask_noobj = test_img < 0.85
    # cvtcolor()函数是一个颜色空间转换函数，可以实现RGB颜色向HSV，HSI等颜色空间转换。也可以转换为灰度图。
    # 掩码图片：[512, 512] 转为：(512, 512, 4)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGBA)
    indexs = np.array(np.where(mask_obj == True))
    indexs = np.stack(indexs, axis=1)
    test_img[indexs] = [0, 0, 200, 100]
    test_img[mask_noobj] = [255, 255, 255, 25]
    mask = Image.fromarray(np.uint8(test_img))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img * 255
    img = Image.fromarray(np.uint8(img))
    mask = mask.convert('RGBA')
    b, g, r, a = mask.split()
    img.paste(mask, (0, 0, 512, 512), mask=a)
    img = np.array(img)
    cv2.imshow('a', img)
    cv2.waitKey(0)
