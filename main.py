import json
from os import path
from dataloader import movieLensDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from model import MF
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
ROOT_PATH = "./data/movielens"
INFO_PATH = path.join(ROOT_PATH, "info.json")
TRAIN_DATA_PATH = path.join(ROOT_PATH, "train.csv")
VALID_DATA_PATH = path.join(ROOT_PATH, "valid.csv")

f = open(INFO_PATH, 'r')
info = json.load(f)
f.close()
user_num = int(info['user_num'])
item_num = int(info['item_num'])

# 数据读取
train_dataset = movieLensDataset(TRAIN_DATA_PATH)
valid_dataset = movieLensDataset(VALID_DATA_PATH)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2048, num_workers=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=2048, num_workers=4)

model = MF(user_num=user_num, item_num=item_num, k_dim=16).cuda()
optimizer = SGD(model.parameters(), lr=5e-6)
loss_func = CrossEntropyLoss()

num_epoch = 100
for i in range(num_epoch):
    # 训练
    train_loss_list, train_l2_loss_list = [], []
    for x, y, r in train_dataloader:
        optimizer.zero_grad()
        r_hat, l2_loss = model.forward(x.cuda(), y.cuda())
        ce_loss = loss_func(r.cuda(), r_hat.squeeze(1))
        loss = ce_loss + 0.01*l2_loss
        loss.backward()
        optimizer.step()
        train_loss_list.append(ce_loss.cpu().item())
        train_l2_loss_list.append(0.0001*l2_loss.cpu().item())

    mse_loss = np.sum(train_loss_list) / train_dataset.__len__()
    l2_loss = np.sum(train_l2_loss_list) / train_dataset.__len__()

    print(f"[train:{i+1}], ce_loss:{mse_loss:.4f}, l2_loss:{l2_loss:.4f}")

    # 验证
    valid_loss_list, valid_l2_loss_list, ACC_list = [], [], []
    r_output = []
    with torch.no_grad():
        for valid_x, valid_y, valid_r in valid_dataloader:
            r_hat, l2_loss = model.forward(valid_x.cuda(), valid_y.cuda())
            ce_loss = loss_func(valid_r.cuda(), r_hat.squeeze(1))
            ACC_list.extend(list((r_hat.squeeze(1).cpu() > 0.5).numpy() == (valid_r.cpu() > 0.5).numpy()))
            valid_loss_list.append(ce_loss.cpu().item())
            valid_l2_loss_list.append(0.0001*l2_loss.cpu().item())
    mse_loss = np.sum(valid_loss_list) / valid_dataset.__len__()
    l2_loss = np.sum(valid_l2_loss_list) / valid_dataset.__len__()
    ACC = np.sum(ACC_list) / valid_dataset.__len__()
    print(f"[valid:{i+1}], ce_loss:{mse_loss:.4f}, l2_loss:{l2_loss:.4f}, ACC:{ACC:.2f}")
