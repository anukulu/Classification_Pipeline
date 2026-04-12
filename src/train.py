import torch
import torch.nn as nn
import torch.optim as optim
from .model import SiameseNetwork
from .preprocess import PrepData
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

net = SiameseNetwork().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.BCELoss()

train_dataset = PrepData(train=True)
test_dataset = PrepData(train=False)
train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=1000)

# print(next(iter(test_loader))[0].shape)
# print(next(iter(test_loader))[0].shape)

net.train()

for epoch in range(2):
    running_loss = 0.0
    for i, (img1, img2, target) in enumerate(train_loader):
        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
        
        optimizer.zero_grad()

        op = net(img1, img2).squeeze()
        loss = criterion(op, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

net.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for img1, img2, target in test_loader:
        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
        op = net(img1, img2).squeeze()
        test_loss += criterion(op, target).sum().item()
        pred = torch.where(op > 0.7, 1, 0)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
