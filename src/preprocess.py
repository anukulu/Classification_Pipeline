import torchvision.datasets as datasets
from torchvision import transforms

data_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


mnist_trainset = datasets.MNIST(
    root="./data/raw",
    train=True,
    download=True,
    transform=data_norm
)

print(mnist_trainset)