# 训练P网络

import nets
import train
import cfg

if __name__ == '__main__':
    net = nets.P_Net()
    trainer = train.Trainer(net, cfg.save_12_params_dir, cfg.img_12_dir)  # 网络、保存参数、训练数据
    trainer.train()  # 调用训练方法
