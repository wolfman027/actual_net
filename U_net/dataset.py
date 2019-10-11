from torch.utils.data import Dataset
import os
import cfg
from PIL import Image
from torchvision import transforms


class MydataSet(Dataset):
    def __init__(self):
        super(MydataSet, self).__init__()
        self.dataset = os.listdir(cfg.train_data_data_dir)
        self.dataset.sort(key=lambda x:int(x.split('.')[0]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = Image.open(os.path.join(cfg.train_data_data_dir, self.dataset[index]))
        lable = Image.open(os.path.join(cfg.train_data_label_dir, self.dataset[index]))
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_data = transform(image)
        lable_data = transform(lable)
        return image_data, lable_data





