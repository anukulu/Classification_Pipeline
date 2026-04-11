from torch import nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(2,2))
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(2,2))
        self.lin1 = nn.Linear(16 * 6 * 6, 256)
        self.lin2 = nn.Linear(256, 128)
        self.op_dim = 64
        self.lin3 = nn.Linear(128, self.op_dim)
    
    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()

        self.fcl = nn.Sequential(
            nn.Linear(2 * self.encoder.op_dim, self.encoder.op_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder.op_dim, 1)
        )

        self.sigmoid = nn.Sigmoid()

        self.encoder.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m , nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            with torch.no_grad():
                m.bias.fill_(0.01)
    
    def forward(self, ip1, ip2):
        op1 = self.encoder(ip1)
        op2 = self.encoder(ip2)

        # this is just a safety net, doesnt do anything in this case
        op1 = op1.view(op1.size()[0], -1)
        op2 = op2.view(op2.size()[0], -1)

        op = torch.cat([op1, op2], dim=1)

        x = self.sigmoid(self.fcl(op))
        return x
    
# model = SiameseNetwork()
# rand_img_1 = torch.randn(8, 1, 28, 28)
# rand_img_2 = torch.randn(8, 1, 28, 28)
# op = model(rand_img_1, rand_img_2)
    

