import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple

class movieLensDataset(Dataset):
    def __init__(self, path: str):
        data = pd.read_csv(path, header=None)
        self.userId = torch.LongTensor(data.iloc[:,0]) - 1
        self.itemID = torch.LongTensor(data.iloc[:,1]) - 1
        self.target = torch.FloatTensor(data.iloc[:,2])


    def __len__(self):
        return len(self.userId)


    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor] :
        return self.userId[index], self.itemID[index], self.target[index]