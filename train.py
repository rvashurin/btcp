from torchvision import datasets, transforms
from torch.utils import data
from model import LeNet
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch

transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_trainset = datasets.FashionMNIST(root='./data', train=True, download=True,
                                       transform=transform)
mnist_testset = datasets.FashionMNIST(root='./data', train=False, download=True,
                                      transform=transform)

train_loader = data.DataLoader(mnist_trainset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(mnist_testset, batch_size=1024, shuffle=True)

model = LeNet()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

for epoch in range(20):
    for (image,label) in train_loader:
        optimizer.zero_grad()
        output = model(image)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target,
                             reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print(f'\nDone with epoch {epoch}')
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
