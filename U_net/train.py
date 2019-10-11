import torch
from torch.utils.data import DataLoader
from Unet import Unet
from dataset import MydataSet
import torch.nn as nn
import cfg

if __name__ == '__main__':
    dataset = MydataSet()
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=4)

    module = Unet(1, 1)
    if torch.cuda.is_available():
        module = module.cuda()

    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam(module.parameters())

    for epoch in range(1000):
        for data, label in dataloader:
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            output = module(data)
            loss = loss_fn(output, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item())

        # 保存模型不要保存一个。。可以保存多个
        torch.save(module.state_dict(), cfg.train_data_params_pt_src)
