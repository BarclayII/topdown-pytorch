import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda")

def normalize(byte_tensor, mean=0.1307, std=0.3081):
    float_tensor = byte_tensor.float() / 256.
    return (float_tensor - mean) / std

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

net = Net().to(device)
batch_size = 64
lr = 1e-2
momentum = 0.5
n_epochs = 12

if __name__ == "__main__":
    from datasets import MNISTScale
    import os
    if not os.path.exists('acc_test_cnn.pt'):
        th.manual_seed(9611)
        train_data = MNISTScale('.', scale=1.0)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        dev_data = MNISTScale('.', mode='valid', scale=1.0)
        dev_loader = DataLoader(dev_data, batch_size=batch_size)
        test_data = MNISTScale('.', mode='test', scale=1.0)
        test_loader = DataLoader(test_data, batch_size=batch_size)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        for epoch in range(n_epochs):
            net.train()
            mean_loss = 0
            for idx, batch in enumerate(train_loader):
                input, target = batch
                input = normalize(input).to(device).unsqueeze(1)
                target = target.to(device).view(-1)
                output = net(input)
                optimizer.zero_grad()
                loss = F.nll_loss(output, target)
                print("Loss at batch {}: {}".format(idx, loss.item()))
                mean_loss += loss.item()
                loss.backward()
                optimizer.step()
            print("Train loss at epoch {}: {}".format(epoch, mean_loss / (1.0 * idx)))
            net.eval()
            hit = 0
            tot = 0
            for batch in dev_loader:
                input, target = batch
                input = normalize(input).to(device).unsqueeze(1)
                target = target.to(device).view(-1)
                output = net(input).max(dim=-1)[1]
                tot += target.size(0)
                hit += (output == target).sum().item()
            print("Accuracy on dev set at epoch {}: {}".format(epoch, 1.0 * hit / tot))

            hit = 0
            tot = 0
            for batch in test_loader:
                input, target = batch
                input = normalize(input).to(device).unsqueeze(1)
                target = target.to(device).view(-1)
                output = net(input).max(dim=-1)[1]
                tot += target.size(0)
                hit += (output == target).sum().item()
            print("Accuracy on test set at epoch {}: {}".format(epoch, 1.0 * hit / tot))

        with open('acc_test_cnn.pt', 'wb') as f:
            torch.save(net, f)
    else:
        import numpy as np
        with open('acc_test_cnn.pt', 'rb') as f:
            net = torch.load(f)
        net.eval()
        for scale in np.linspace(0.2, 1, 21):
            test_data = MNISTScale('.', mode='test', scale=scale)
            test_loader = DataLoader(test_data, batch_size=batch_size)
            hit = 0
            tot = 0
            for batch in test_loader:
                input, target = batch
                input = normalize(input).to(device).unsqueeze(1)
                target = target.to(device).view(-1)
                output = net(input).max(dim=-1)[1]
                tot += target.size(0)
                hit += (output == target).sum().item()
            print(1.0 * hit / tot)
"""
0.1135,
0.1135,
0.1135,
0.1909,
0.0891,
0.173,
0.0912,
0.0832,
0.1241,
0.1316,
0.477,
0.5984,
0.5168,
0.7952,
0.7973,
0.9157,
0.8441,
0.7692,
0.9677,
0.9427,
0.984
"""
