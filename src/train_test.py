import torch
import torch.nn as nn
import torch.optim as optim
from .model import SiameseNetwork
from .preprocess import PrepData
from torch.utils.data import DataLoader
import os
from pathlib import Path

class TrainTest:
    def __init__(self,
                train_batch_size : int,
                test_batch_size : int,
                learning_rate : float,
                sim_threshold : float = 0.7
        ):
        self.sim_threshold = sim_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SiameseNetwork().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

        self.train_loader = DataLoader(PrepData(train=True), batch_size=train_batch_size)
        self.test_loader = DataLoader(PrepData(train=False), batch_size=test_batch_size)


    def train_model(self):
        self.net.train()

        for epoch in range(2):
            running_loss = 0.0
            for i, (img1, img2, target) in enumerate(self.train_loader):
                img1, img2, target = img1.to(self.device), img2.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()

                op = self.net(img1, img2).squeeze()
                loss = self.criterion(op, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 0:
                    print(f'Epoch :{epoch + 1}, Step : {i + 1:5d}, Loss: {running_loss / 10:.3f}')
                    running_loss = 0.0
        print("Model has been trained.")


    def test_model(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for img1, img2, target in self.test_loader:
                img1, img2, target = img1.to(self.device), img2.to(self.device), target.to(self.device)
                op = self.net(img1, img2).squeeze()
                test_loss += self.criterion(op, target).sum().item()
                pred = torch.where(op > self.sim_threshold, 1, 0)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))
    
    def save_or_load(self, model_name : str = "trained_model.pth", save : bool = True):
        curr_file = Path(__file__).resolve()
        root = curr_file.parent.parent
        model_path = root / "models" / model_name
        if save:
            torch.save(self.net.state_dict(), model_path)
            print(f"Model has been successfully saved to {model_path}")
        else:
            if model_path.exists():
                try:
                    self.net.load_state_dict(torch.load(model_path))
                except Exception as e:
                    raise ValueError(f"Could not load the model. {e}")

# trainer = TrainTest(train_batch_size=64, test_batch_size=1000, learning_rate=0.001)

# if training and testing use this
# trainer.train_model()
# trainer.test_model()
# trainer.save_or_load(save=True)

# if only testing use this
# trainer.save_or_load(save=False)
# trainer.test_model()