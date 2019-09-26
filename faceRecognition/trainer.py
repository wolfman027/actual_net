from dateset import FaceDataset
import faceNet
import cfg
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

net = faceNet.FaceNet()
if torch.cuda.is_available():
    net = net.cuda()
net.train()

loss_fn = nn.NLLLoss()

opt = torch.optim.Adam(net.parameters())

dataset = FaceDataset(cfg.train_face_img_main_dir)
dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

for epoch in range(100000):
    for xs, ys in dataloader:
        if torch.cuda.is_available():
            xs = xs.cuda()
            ys = ys.cuda()
        feature, cls = net(xs)
        print(torch.argmax(cls, dim=1), ys)
        # print(torch.log(cls))
        loss = loss_fn(torch.log(cls), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
    if epoch % 100 == 0:
        torch.save(net.state_dict(), cfg.net_params_dir)
























