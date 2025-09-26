from torch import nn, save, load # Our neural network
from torch.optim import Adam # Our optimization model
from torch.utils.data import DataLoader # Needed to load our datasets
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get our data
train = datasets.MNIST(root = "data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

# Image Classifier NN
class ImageClassifier(nn.Module):
    def __init__(self):
        print('clf init')
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)
    
# Instance of the neural network, loss & optimizer

clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(), lr=0.15)
loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    print('starting epochs')
    for epoch in range(3):
        for batch in dataset:
            print(batch)
            X, y = batch
            X, y = X.to('cpu'), y.to('cpu')
            yhat = clf(X)
            loss = loss_fn(yhat, y)
            
            # Apply backprop

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch: {epoch} loss is {loss.item()}")

    with open('modelstate.pt', 'wb') as f:
        save(clf.state_dict(), f)