# 训练神经网络
import dataset
from darknet531 import MainNet
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import cfg
import utils
from torch.utils.tensorboard import SummaryWriter


conf_loss_fn = nn.BCEWithLogitsLoss()       # 置信度：二值交叉熵
center_loss_fn = nn.BCEWithLogitsLoss()     # 中心点：二值交叉熵
wh_loss_fn = nn.MSELoss()                   # 宽高：均方差
# cls_loss_fn = nn.CrossEntropyLoss()         # 分类：交叉熵
cls_loss_fn = nn.MSELoss()


def loss_fn(output, target, alpha):
    # 把NCHW - > NHWC
    output = output.permute(0,2,3,1)
    # 把NHWC->NHW3*15
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3,-1)
    target = target.to(utils.getDevice())
    # 负样本的时候只需要计算置信度损失
    mask_noobj = target[..., 4] <= 0.1
    output_noobj, target_noobj = output[mask_noobj], target[mask_noobj]
    loss_noobj = conf_loss_fn(output_noobj[:, 4], target_noobj[:, 4])
    # 损失分为两部分，一部分为有样本的格子，一部分为没有样本的各自
    mask_obj = target[..., 4] > 0.1
    output_obj, target_obj = output[mask_obj], target[mask_obj]
    if output_obj.size(0) > 0:
        loss_obj_conf = conf_loss_fn(output_obj[:, 4], target_obj[:, 4])     # 置信度损失
        loss_obj_center = center_loss_fn(output_obj[:, 0:2], target_obj[:, 0:2])     # 中心点偏移量损失
        loss_obj_wh = wh_loss_fn(output_obj[:, 2:4], target_obj[:, 2:4])     # 宽高
        # loss_obj_cls = cls_loss_fn(output_obj[:,5:], target_obj[:,5].long())
        target_obj_cls = target_obj[:, 5].unsqueeze(0).reshape(-1, 1)
        target_obj_cls_one_hot = torch.zeros(target_obj_cls.size(0), cfg.class_num, device='cuda').scatter_(1, target_obj_cls.long(), 1)
        # print(target_obj_cls_one_hot.shape)
        # print(output_obj[:, 5:].shape)

        loss_obj_cls = cls_loss_fn(output_obj[:, 5:], target_obj_cls_one_hot)    # 改为用MSE损失
        loss_obj = loss_obj_conf + loss_obj_center + loss_obj_wh + loss_obj_cls
        return alpha * loss_obj + (1 - alpha) * loss_noobj
    else:
        return loss_noobj


if __name__ == '__main__':
    myDataset = dataset.CocoDataset()
    # drop_last 批次不够的时候是否丢掉
    train_loader = DataLoader(myDataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
    # 创建网络
    net = MainNet(cfg.class_num).to(utils.getDevice())
    # 加载权重
    # net.load_state_dict(torch.load('data/params/ckpt-185.pt'))
    # 开始训练
    net.train()
    # 增加观察参数对象
    # summaryWriter = SummaryWriter()

    # 定义优化器
    opt = torch.optim.Adam(net.parameters())

    for epoch in range(10000):
        for i, (target_13, target_26, target_52, img_data) in enumerate(train_loader):
            output_13, output_26, output_52 = net(img_data.to(utils.getDevice()))
            loss_13 = loss_fn(output_13, target_13, 0.9)
            loss_26 = loss_fn(output_26, target_26, 0.9)
            loss_52 = loss_fn(output_52, target_52, 0.9)
            loss = loss_13 + loss_26 + loss_52
            opt.zero_grad()
            loss.backward()
            opt.step()
            print("epoch：{},no：{}, loss：{}".format(epoch, i, loss.item()))
            # summaryWriter.add_histogram("w1", net.trunk_52[0].sub_model[0].weight.data, global_step=epoch)
            # summaryWriter.add_scalar("loss", loss, global_step=epoch)
        if epoch == 5000:
            torch.save(net.state_dict(), "data/params/ckpt-{}.pt".format(epoch))
    # summaryWriter.close()
