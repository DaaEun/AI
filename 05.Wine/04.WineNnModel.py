import csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
torch.set_printoptions(edgeitems=2, precision=2, linewidth=75)


# load a wine data
wine_path = "tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)

# col_list = next(csv.reader(open(wine_path), delimiter=';'))
# print(col_list)

# wineq = torch.from_numpy(wineq_numpy)
# print(wineq.shape, wineq.dtype)

# splitting a dataset
wine_samples = wineq_numpy.shape[0]
wine_val = int(0.2 * wine_samples)

shuffled_indices = torch.randperm(wine_samples)
train_indices = shuffled_indices[:-wine_val]
val_indices = shuffled_indices[-wine_val:]

wine_train = wine_samples[train_indices]
wine_val = wine_samples[val_indices]

# the pytorch nn module
leakyReLU_model = nn.LeakyReLU(1, 1) 
leakyReLU_model(wine_val)

# random하게 w, b 초기화
leakyReLU_model.weight
leakyReLU_model.bias

# batch inputs
x = torch.ones(12, 1)
leakyReLU_model(x)

# loss_function
loss_fn = nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(leakyReLU_model.parameters(), lr=0.00001, weight_decay=0.01, amsgrad=True)

# training_loop
def training_loop(n_epochs, optimizer, model, loss_fn, wine_train, wine_val):
    
    for epoch in range(1, n_epochs + 1):
    
        t_p_train = model(wine_train) # 변경
        loss_train = loss_fn(t_p_train, t_c_train)

        t_p_val = model(t_u_val) # 변경
        loss_val = loss_fn(t_p_val, t_c_val)
        
        optimizer.zero_grad()
        loss_train.backward()   # backward 진행 -> grad 계산
        optimizer.step()        # params 업데이트        

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                f" Validation loss {loss_val.item():.4f}")
