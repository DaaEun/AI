import numpy as np
import torch
torch.set_printoptions(edgeitems=2)

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
t_un = 0.1 * t_u    # Normalizing

# model 클래스
# x : input / t_u
# b : bias
# y = w*x + b
def model(t_u, w, b):
    return w * t_u + b

# loss fucntion 클래스
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()     # MSE


###############################
#### Optimizers a la carte ####
###############################

import torch.optim as optim

dir(optim)
"""
['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 
'NAdam', 'Optimizer', 'RAdam', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam', 
'__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', 
'__package__', '__path__', '__spec__', '_functional', '_multi_tensor', 'lr_scheduler', 'swa_utils']
"""


############################################
#### Using a gradient descent optimizer ####
############################################

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5    # lr 작게
optimizer = optim.SGD([params], lr=learning_rate)   # SGD : 가장 기본 알고리즘

t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)
loss.backward()

optimizer.step()    # params 업데이트

params
"""
tensor([ 9.5483e-01, -8.2600e-04], requires_grad=True)
"""


################################
#### Zero out the gradients ####
################################

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)
optimizer.zero_grad() # g = 0 초기화 / 반드시 기입!!
loss.backward()

optimizer.step()

params
"""
tensor([1.7761, 0.1064], requires_grad=True)
"""


###########################
###### Training loop ######
###########################

def training_loop(n_epochs, optimizer, params, t_u, t_c):   # optimizer 추가
    for epoch in range(1, n_epochs + 1):

        t_p = model(t_u, *params)   # forward
        loss = loss_fn(t_p, t_c)    # forward

        optimizer.zero_grad()   # 추가
        loss.backward()
        optimizer.step()        # 추가

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params


##############################
########## Training ##########
##############################

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)  

training_params = training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    params = params,
    t_u = t_un,  
    t_c = t_c)
print(training_params)
"""
Epoch 500, Loss 7.860115
Epoch 1000, Loss 3.828538
Epoch 1500, Loss 3.092191
Epoch 2000, Loss 2.957698
Epoch 2500, Loss 2.933134
Epoch 3000, Loss 2.928648
Epoch 3500, Loss 2.927830
Epoch 4000, Loss 2.927679
Epoch 4500, Loss 2.927652
Epoch 5000, Loss 2.927647
tensor([  5.3671, -17.3012], requires_grad=True)
"""


################################
######## Adam optimizer ########
################################

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-1    # 변경
optimizer = optim.Adam([params], lr=learning_rate)  # Adam 변경

training_params = training_loop(
    n_epochs = 2000,    # 횟수감소 (변경)
    optimizer = optimizer,
    params = params,
    t_u = t_u,          # Normalizing X (변경)
    t_c = t_c)
print(training_params)
"""
Epoch 500, Loss 7.612900
Epoch 1000, Loss 3.086700
Epoch 1500, Loss 2.928579
Epoch 2000, Loss 2.927644
tensor([  0.5367, -17.3021], requires_grad=True)
"""


#####################################
######## Splitting a dataset ########
#####################################

n_samples = t_u.shape[0]        # 0번째 차원 = 입력 수 n
n_val = int(0.2 * n_samples)    # 전체 n개 중 20% = validation set

shuffled_indices = torch.randperm(n_samples)    # index 뒤섞기

train_indices = shuffled_indices[:-n_val]           
val_indices = shuffled_indices[-n_val:]

print(train_indices, val_indices)
"""
tensor([ 3,  8,  0,  6,  7, 10,  9,  2,  4]) tensor([1, 5])
"""

# train_set
train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

# validation_set
val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u    # Normalizing
val_t_un = 0.1 * val_t_u        # Normalizing

# training_loop with validation set
def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):

        # train_set forward
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)

        # validation_set forward                     
        val_t_p = model(val_t_u, *params) 
        val_loss = loss_fn(val_t_p, val_t_c)

        optimizer.zero_grad()   
        train_loss.backward()   # train_set brackward
        optimizer.step()       

        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f}," 
                    f" Validation loss {val_loss.item():.4f}")

    return params

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)   # SGD

training_params = training_loop(
    n_epochs = 3000,    
    optimizer = optimizer,
    params = params,
    train_t_u = train_t_un,
    val_t_u = val_t_un,
    train_t_c = train_t_c,
    val_t_c = val_t_c)
print(training_params)
"""
Epoch 1, Training loss 89.2897, Validation loss 40.2001
Epoch 2, Training loss 42.9957, Validation loss 7.5859
Epoch 3, Training loss 35.9978, Validation loss 5.1593
Epoch 500, Training loss 7.4615, Validation loss 2.4141
Epoch 1000, Training loss 3.8338, Validation loss 1.6143
Epoch 1500, Training loss 3.3591, Validation loss 1.4178
Epoch 2000, Training loss 3.2970, Validation loss 1.3589
Epoch 2500, Training loss 3.2889, Validation loss 1.3391
Epoch 3000, Training loss 3.2878, Validation loss 1.3322
tensor([  5.3320, -17.1560], requires_grad=True)
"""


#######################################
######## with torch.no_grad(): ########
#######################################
# gradient를 계산하지 않는 경우

def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):

    for epoch in range(1, n_epochs + 1):

        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)

        with torch.no_grad(): # 추가
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            # assert : check! 조건을 만족하면 동작, 아니면 stop 
            assert val_loss.requires_grad == False  
            
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f}," 
                    f" Validation loss {val_loss.item():.4f}")

    return params

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)   # SGD

training_params = training_loop(
    n_epochs = 3000,    
    optimizer = optimizer,
    params = params,
    train_t_u = train_t_un,
    val_t_u = val_t_un,
    train_t_c = train_t_c,
    val_t_c = val_t_c)
print(training_params)
"""
Epoch 1, Training loss 93.9718, Validation loss 19.1309
Epoch 2, Training loss 37.0213, Validation loss 20.6740
Epoch 3, Training loss 29.4933, Validation loss 30.6736
Epoch 500, Training loss 7.2228, Validation loss 13.3709
Epoch 1000, Training loss 3.5706, Validation loss 7.1267
Epoch 1500, Training loss 2.9480, Validation loss 5.2825
Epoch 2000, Training loss 2.8418, Validation loss 4.6461
Epoch 2500, Training loss 2.8237, Validation loss 4.4047
Epoch 3000, Training loss 2.8207, Validation loss 4.3087
tensor([  5.2846, -16.4926], requires_grad=True)
"""


####################################
##### set_grad_enabled context #####
####################################
# gradient 계산 할거니 말거니 함수

# is_train : boolean
def calc_forward(t_u, t_c, is_train):
    # is_train = True 이면, grad 계산 하고,
    # is_train = False 이면, 계산 X
    with torch.set_grad_enabled(is_train):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
    return loss