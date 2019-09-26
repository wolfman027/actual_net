from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import cfg

# 缩放图片
data_for = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor()
])


# 人脸数据
class FaceDataset(Dataset):
    def __init__(self, main_dir):
        super(FaceDataset, self).__init__()
        # 保存的是：[人脸图片地址，人脸的类别]
        self.dataset = []
        for face_dir in os.listdir(main_dir):
            for face_filename in os.listdir(os.path.join(main_dir, face_dir)):
                self.dataset.append([os.path.join(main_dir, face_dir, face_filename), int(face_dir)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img_data = data_for(Image.open(data[0]))
        return img_data, data[1]


# 测试代码写的是否正确
if __name__ == '__main__':
    faceDataset = FaceDataset(cfg.face_img_main_dir)
    print(faceDataset[1])










