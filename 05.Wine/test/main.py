import csv
import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader 

random.seed(0)

class WineDataset(Dataset):
    def __init__(self):
        wine_path = "/data/winequality-white.csv"
        items = np.loadtxt(wine_path, dtype=np.float32, delimiter=';', skiprows=1)
        num_data = np.shape(items)[0]

        indices = [i for i in range(num_data)]
        random.shuffle(indices)

        train_set_size = int(len(indices) * 0.9)

        self.train_items = items[indices[:train_set_size], :] 
        self.test_items = items[indices[train_set_size:], :] 
        self.items = self.train_items

    def __len__(self):
        return np.shape(self.items)[0]

    def __getitem__(self, index):
        return self.items[index][:-1], self.items[index][-1]

    def set_test_mode(self):
        self.items = self.test_items

    def set_train_mode(self):
        self.items = self.train_items


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.leakyReLU = nn.LeakyReLU()

        self.fc1 = nn.Linear(11, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leakyReLU(x)
        x = self.fc2(x)
        x = self.leakyReLU(x)
        x = self.fc3(x)
        x = self.leakyReLU(x)
        x = self.fc4(x)
        x = self.leakyReLU(x)
        return x

# data loader
dataset = WineDataset()
data_loader = DataLoader(
    dataset,
    shuffle=True,
    num_workers=8,
    batch_size=8,
    pin_memory=True
)

# model
model = Model().to('cuda')

# criterion
criterion = torch.nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01, amsgrad=True)

# train
num_epoch = 50
train_loss = []
test_loss = []
for epoch in tqdm(range(num_epoch), ncols=90):
    # train
    loss_sum = 0
    count = 0
    dataset.set_train_mode()
    model.train()
    for x, label in tqdm(data_loader, desc='train', ncols=90):
        x = x.to('cuda')
        label = label.to('cuda')

        # output
        output = model(x)

        # loss
        loss = criterion(output, label) 
        loss_sum += loss.item()
        count += 1

        # backpropagation
        loss.backward()
        optimizer.step()

    print(epoch, "loss: ", loss_sum / count)
    train_loss.append(loss_sum / count)

    # test
    dataset.set_test_mode()
    loss_sum = 0
    count = 0
    model.eval()
    for x, label in tqdm(data_loader, desc='test', ncols=90):
        with torch.set_grad_enabled(False):
            x = x.to('cufda')
            label = label.to('cuda')

            # output
            output = model(x)

            # loss
            loss = criterion(output, label) 
            loss_sum += loss.item()
            count += 1

    print(epoch, "test_loss: ", loss_sum / count)
    test_loss.append(loss_sum / count)

plt.plot([i for i in range(num_epoch)], train_loss)
plt.plot([i for i in range(num_epoch)], test_loss)
plt.legend(['train', 'test'])
plt.savefig('result', dpi=300)