import torch
from torch import nn
from typing import Tuple

class MF(nn.Module):
    def __init__(self, user_num, item_num, k_dim):
        super().__init__()
        self.users_embedding = nn.Embedding(user_num, k_dim)
        self.users_bias = nn.Embedding(user_num, 1)
        self.items_embedding = nn.Embedding(item_num, k_dim)
        self.items_bisa = nn.Embedding(item_num, 1)
        self.bias = nn.Embedding(1, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # batch计算, len(users) = len(items) = batch
        batch = users.shape[0]
        users_embed = self.users_embedding.weight[users] # (batch * dim)
        users_bias = self.users_bias.weight[users] # (batch * 1)
        items_embed = self.items_embedding.weight[items] # (batch * dim)
        items_bias = self.items_bisa.weight[items] # (batch * dim)

        # output = eu*ei + bu + bi + b -> (batch, 1)
        output = self.output_activation(torch.sum(torch.mul(users_embed, items_embed)) + users_bias + items_bias + self.bias.weight.repeat(batch, 1))
        # L2损失函数
        L2_loss = torch.sum(users_embed.pow(2)) + torch.sum(items_embed.pow(2)) + torch.sum(users_bias.pow(2)) + torch.sum(items_bias.pow(2))

        return output, L2_loss
