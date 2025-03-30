import torch
from torch.utils.data import random_split

data = torch.load('./data/inj_cora.pt')

from copy import deepcopy

train_data = deepcopy(data)
test_data = deepcopy(data)

train_data.test_mask = None
test_data.train_mask = None

torch.save(train_data, './data/inj_cora_train.pt')
torch.save(test_data, './data/inj_cora_test.pt')
