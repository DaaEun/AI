import csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from matplotlib import pyplot as plt
torch.set_printoptions(edgeitems=2, precision=2, linewidth=75)

# load a wine data
wine_path = "05.WineQuality/red_wine_data_analysis.CSV"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=",", skiprows=1)
wineq = torch.from_numpy(wineq_numpy)

# data : volatile acidity, sulphates, alcohol  속성에 해당하는 데이터 -> 3
data = wineq[:, :-1]    
target = wineq[:, -1]   # target : quality

# splitting a dataset
wine_samples = data.shape[0]
wine_val = int(0.2 * wine_samples)

shuffled_indices = torch.randperm(wine_samples)
train_indices = shuffled_indices[:-wine_val]
val_indices = shuffled_indices[-wine_val:]

data_train = data[train_indices]    # train set
target_train = target[train_indices]
data_val = data[val_indices]        # validation set
target_val = target[val_indices]

# model : the pytorch nn module
seq_model = nn.Sequential(OrderedDict([
    ('hidden1_linear', nn.Linear(3, 64)),
    ('hidden1_activation', nn.LeakyReLU()),
    ('hidden2_linear', nn.Linear(64, 128)),
    ('hidden2_activation', nn.LeakyReLU()),
    # ('hidden3_linear', nn.Linear(128, 64)),
    # ('hidden3_activation', nn.LeakyReLU()),
    ('output_linear', nn.Linear(128, 1))
]))

# loss_function
loss_fn = nn.MSELoss()

# optimizer
optimizer = optim.Adam(seq_model.parameters(), lr=1e-2, weight_decay=0.01, amsgrad=True)

# n_epochs
n_epochs = 50

# loss value
train_loss = []
val_loss = []

# training_loop
def training_loop(n_epochs, optimizer, model, loss_fn, data_train, data_val, target_train, target_val):
    
    for epoch in range(1, n_epochs + 1):
    
        y_train = model(data_train)
        loss_train = loss_fn(y_train, target_train)

        y_val = model(data_val) 
        loss_val = loss_fn(y_val, target_val)
        
        optimizer.zero_grad()
        loss_train.backward()  
        optimizer.step()       

        train_loss.append(loss_train.item())
        val_loss.append(loss_val.item())

        # monitoring
        # print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
        #         f" Validation loss {loss_val.item():.4f}")
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                f" Validation loss {loss_val.item():.4f}")

# training 
training_loop(
    n_epochs = n_epochs, 
    optimizer = optimizer,
    model = seq_model,
    loss_fn = loss_fn, 
    data_train = data_train, 
    data_val = data_val, 
    target_train = target_train, 
    target_val = target_val)

# print png
fig = plt.figure(dpi=600)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot([i for i in range(1, n_epochs + 1)], train_loss)
plt.plot([i for i in range(1, n_epochs + 1)], val_loss)
plt.legend(['train', 'validation'])
# plt.savefig("wine_train.png", format="png")
plt.savefig("05.WineQuality/red_wine_data_analysis_img/hidden2#64#128_epoch50.png", format="png")