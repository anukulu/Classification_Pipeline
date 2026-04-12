import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import random
import torch

class PrepData(Dataset):

    def __init__(self, **kwargs):
        super().__init__()
 
        data_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST(
            root="./data/raw",
            train=kwargs.get("train"),
            download=True,
            transform=data_norm
        )

        self.dataset = dataset.data.unsqueeze(1).clone()
        self.targets = dataset.targets.clone()

        self.group_examples()

    def group_examples(self):

        y = np.array(self.targets, dtype=None, copy=None)

        self.group = {}
        for i in range(10):
            self.group[i] = np.where((y == i))[0]
        
    # we pick a random class, if the index is even, choose a positive example
    # for the example from a choice otherwise just choose a negative example
    def __getitem__(self, index):

        rand_class = random.randint(0, 9)
        first_eg_ind = random.choice(self.group[rand_class])
        first_eg = self.dataset[first_eg_ind].clone().float()

        if index % 2 == 0:
            second_eg_ind = random.choice(self.group[rand_class])
            while first_eg_ind == second_eg_ind:
                second_eg_ind = random.choice(self.group[rand_class])
            
            second_eg = self.dataset[second_eg_ind].clone().float()
            target = torch.tensor(1, dtype=torch.float)

        else:
            another_class = random.randint(0, 9)
            while another_class == rand_class:
                another_class = random.randint(0, 9)
            
            second_eg_ind = random.choice(self.group[another_class])
            second_eg = self.dataset[second_eg_ind].clone().float()
            target = torch.tensor(0, dtype=torch.float) 
        
        return first_eg, second_eg, target
    

    def __len__(self):
        return len(self.dataset)

# dataset = PrepData()
# loader = DataLoader(dataset, batch_size=8)

# print(next(iter(loader))[0].shape)