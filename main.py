from torch import nn # Our neural network
from torch.optim import Adam # Our optimization model
from torch.utils.data import DataLoader # Needed to load our datasets
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MINST(root = "data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)


class ImageClassifier(nn.Module):
    def _init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU,
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU,
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU,
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)
