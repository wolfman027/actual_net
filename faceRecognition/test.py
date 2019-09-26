from faceNet import FaceNet, compare
import torch
from torchvision import transforms
from PIL import Image
import torch.jit as jit
import cfg
import os

# 缩放图片
data_for = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
])

# 使用

net = FaceNet().cuda()
net.load_state_dict(torch.load(cfg.net_params_dir))
net.eval()

person1 = data_for(Image.open(os.path.join(cfg.train_face_img_main_dir, '0/000_0.bmp')))
person1 = person1.cuda()
person1_feature = net.encode(person1[None, ...])  # 得到person1得特征 person1[None, ...] 这样写是加一个维度得简单方法

person2 = data_for(Image.open(os.path.join(cfg.train_face_img_main_dir, '1/001_1.bmp')))
person2 = person2.cuda()
person2_feature = net.encode(person2[None, ...])  # 得到person2得特征

siam = compare(person1_feature, person2_feature)  # 计算 person1 与 person2 得余弦相似度
print(siam)


# 把模型和参数进行打包，以便C++或PYTHON调用
# x = torch.Tensor(1, 3, 112, 112)
# traced_script_module = jit.trace(net, x)
# traced_script_module.save("model.cpt")  # 最后把打包好的model.cpt交给c++程序员
