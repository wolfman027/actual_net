# 创建训练器 —— 供三个网络训练用
import torch.nn as nn
import torch
import os
import dataset
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, net, save_params_path, dataset_path, isCuda=True):
        self.net = net
        self.save_params_path = save_params_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda

        if self.isCuda:
            self.net.cuda()

        # 创建损失函数
        self.conf_loss_fn = nn.BCELoss()
        self.face_loss_fn = nn.MSELoss()
        self.facial_loss_fn = nn.MSELoss()

        # 创建优化器
        self.opt = torch.optim.Adam(self.net.parameters())
        # 恢复网络训练---加载模型参数，继续训练
        if os.path.exists(self.save_params_path):  # 如果文件存在，接着继续训练
            net.load_state_dict(torch.load(self.save_params_path))

    # 训练方法
    def train(self):
        # 数据集
        faceDataset = dataset.FaceDataset(self.dataset_path)
        faceDataloader = DataLoader(faceDataset, batch_size=512, shuffle=True, num_workers=4, drop_last=True)
        j = 0
        while True:
            for i, (img_data, cond_target, face_offset_target, facial_offset_target) in enumerate(faceDataloader):
                if self.isCuda:
                    img_data = img_data.cuda()
                    cond_target = cond_target.cuda()
                    face_offset_target = face_offset_target.cuda()
                    facial_offset_target = facial_offset_target.cuda()

                # 网络输出
                cond_output_, offset_face_output_, offset_facial_output_ = self.net(img_data)
                cond_output = cond_output_.view(-1, 1)
                offset_face_output = offset_face_output_.view(-1, 4)
                offset_facial_output = offset_facial_output_.view(-1, 10)

                # 计算分类的损失----置信度
                # 对置信度小于2的正样本（1）和负样本（0）进行掩码; ★部分样本（2）不参与损失计算；符合条件的返回1，不符合条件的返回0
                cond_target_mask = torch.lt(cond_target, 2)
                # 对“标签”中置信度小于2的选择掩码，返回符合条件的结果
                target_cond = torch.masked_select(cond_target, cond_target_mask)
                # 预测的“标签”进掩码，返回符合条件的结果
                cond_output = torch.masked_select(cond_output, cond_target_mask)
                cond_loss = self.conf_loss_fn(cond_output, target_cond)  # 对置信度做损失

                # 计算bound回归的损失----偏移量
                # 对置信度大于0的标签，进行掩码；★负样本不参与计算,负样本没偏移量;[512,1]
                face_offset_mask = torch.gt(cond_target, 0)
                # 选出非负样本的索引；[244]
                face_offset_index = torch.nonzero(face_offset_mask)[:, 0]
                # 标签里饿偏移量；[244,4]
                target_face_offset = face_offset_target[face_offset_index]
                # 输出的偏移量；[244,4]
                output_face_offset = offset_face_output[face_offset_index]
                # 偏移量损失
                offset_face_loss = self.face_loss_fn(output_face_offset, target_face_offset)  # 偏移量损失

                # 计算五官位置----偏移量
                facial_offset_mask = torch.gt(cond_target, 0)
                facial_offset_index = torch.nonzero(facial_offset_mask)[:, 0]
                target_facial_offset = facial_offset_target[facial_offset_index]
                output_facial_offset = offset_facial_output[facial_offset_index]
                offset_facial_loss = self.facial_loss_fn(output_facial_offset, target_facial_offset)

                # 总损失
                loss = cond_loss + offset_face_loss + offset_facial_loss
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # 输出损失：loss-->gpu-->cup（变量）-->tensor-->array
                print("i=", i, "loss:", loss.cpu().data.numpy(), " cond_loss:", cond_loss.cpu().data.numpy(),
                      " face_offset_loss", offset_face_loss.cpu().data.numpy(),
                      " facial_offset_loss", offset_facial_loss.cpu().data.numpy())
                # 保存
                if int((i + 1) % 100) == 0:
                    torch.save(self.net.state_dict(), self.save_params_path)  # state_dict保存网络参数，save_path参数保存路径
                    print("save success")
                j +=1

            if int((j + 1) % 10) == 0:
                break
