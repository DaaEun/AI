# Linear model vs Neuron Network 
# -> nonlinear activate function 존재 유무
# Linear model : X
# Neuron Network : O

import numpy as np
import torch
import torch.optim as optim
torch.set_printoptions(edgeitems=2, linewidth=75)

# unsqueeze() : 차원 추가 <-> squeeze
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]    # 1차원 tensor, 1차원 array
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]    # 1차원 tensor, 1차원 array
t_c = torch.tensor(t_c).unsqueeze(1) # 차원 1 추가
t_u = torch.tensor(t_u).unsqueeze(1) # 차원 1 추가

print(t_u.shape)
"""
torch.Size([11, 1])
"""

# splitting a dataset
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

print(train_indices, val_indices)
"""
tensor([4, 1, 5, 0, 6, 2, 3, 9, 7]) tensor([10,  8])
"""

t_u_train = t_u[train_indices]
t_c_train = t_c[train_indices]

t_u_val = t_u[val_indices]
t_c_val = t_c[val_indices]

t_un_train = 0.1 * t_u_train
t_un_val = 0.1 * t_u_val


#######################################
######## The Pytorch nn module ########
#######################################

import torch.nn as nn

# nn.Linear(input_size, outpit_size, bias(default True))
linear_model = nn.Linear(1, 1) 
print(linear_model(t_un_val))
"""
tensor([[3.6385], [2.4330]], grad_fn=<AddmmBackward0>)
"""

# random하게 w, b 초기화
print(linear_model.weight)
"""
Parameter containing:
tensor([[0.8798]], requires_grad=True)
"""
# [0.8798] : w
print(linear_model.bias)
"""
Parameter containing:
tensor([0.3643], requires_grad=True)
"""
# [0.3643] : b


#################################
######## Batching inputs ########
#################################

# 입력 1개가 10번 수행 = batch_size
# x_size = 10 * 1 = batch_size * Nin
# torch.ones(n, m) 1로 채워진 n×m 텐서 생성
x = torch.ones(10, 1)
print(linear_model(x))
"""
tensor([[-0.2128],
        [-0.2128],
        [-0.2128],
        [-0.2128],
        [-0.2128],
        [-0.2128],
        [-0.2128],
        [-0.2128],
        [-0.2128],
        [-0.2128]], grad_fn=<AddmmBackward0>)
"""
# 결과 또한 10개 출력


###############################
######## Training Code ########
###############################

linear_model = nn.Linear(1, 1)
# linear_model.parameters() : linear 모델안에 있는 params -> w, b
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

print(linear_model.parameters())
"""
<generator object Module.parameters at 0x0000019E8D188A50>
"""
print(list(linear_model.parameters()))
"""
[Parameter containing:
tensor([[0.3725]], requires_grad=True), 
Parameter containing:
tensor([0.8037], requires_grad=True)]
"""
# [0.3725] : w / [0.8037] : b
# 이전 강의에서는 모델이라는 함수를 제작했으나, 이제부터는 이미 만들어진 모델 사용

# training_loop
def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val): # model 추가
    
    for epoch in range(1, n_epochs + 1):
    
        t_p_train = model(t_u_train) # 변경
        loss_train = loss_fn(t_p_train, t_c_train)

        t_p_val = model(t_u_val) # 변경
        loss_val = loss_fn(t_p_val, t_c_val)
        
        optimizer.zero_grad()
        loss_train.backward()   # backward 진행 -> grad 계산
        optimizer.step()        # params 업데이트        

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                f" Validation loss {loss_val.item():.4f}")

# loss_function
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

# training 
training_loop(
    n_epochs = 3000, 
    optimizer = optimizer,
    model = linear_model,   # 변경
    loss_fn = loss_fn,
    t_u_train = t_un_train,
    t_u_val = t_un_val, 
    t_c_train = t_c_train,
    t_c_val = t_c_val)

print()
print(linear_model.weight)
print(linear_model.bias)
"""
Epoch 1, Training loss 116.9345, Validation loss 62.4656
Epoch 1000, Training loss 3.6761, Validation loss 1.9048
Epoch 2000, Training loss 2.8077, Validation loss 4.0198
Epoch 3000, Training loss 2.7798, Validation loss 4.5247

Parameter containing:
tensor([[5.4924]], requires_grad=True)
Parameter containing:
tensor([-18.3104], requires_grad=True)
"""


#######################################################
#### nn.MSELoss (MSE stands for Mean Square Error) ####
#######################################################

# training_2
training_loop(
    n_epochs = 3000, 
    optimizer = optimizer,
    model = linear_model,  
    loss_fn = nn.MSELoss(), # 변경
    t_u_train = t_un_train,
    t_u_val = t_un_val, 
    t_c_train = t_c_train,
    t_c_val = t_c_val)

print()
print(linear_model.weight)
print(linear_model.bias)
"""
Epoch 1, Training loss 3.1084, Validation loss 3.4086
Epoch 1000, Training loss 3.0055, Validation loss 3.1391
Epoch 2000, Training loss 2.9843, Validation loss 3.2117
Epoch 3000, Training loss 2.9799, Validation loss 3.2850

Parameter containing:
tensor([[5.4111]], requires_grad=True)
Parameter containing:
tensor([-17.8501], requires_grad=True)
"""


####################################
#### Replacing the linear model ####
####################################

# Sequential() : 가장 기본적인 model
seq_model = nn.Sequential(
            nn.Linear(1, 13), 
            nn.Tanh(),      # activate fuction
            nn.Linear(13, 1)) 
print(seq_model)
"""
Sequential(
  (0): Linear(in_features=1, out_features=13, bias=True)
  (1): Tanh()
  (2): Linear(in_features=13, out_features=1, bias=True)
)
"""


###################################
#### Inspecting the parameters ####
###################################

# model안 params에 대한 shape 확인
print([param.shape for param in seq_model.parameters()])
"""
[torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
"""

# named_params
for name, param in seq_model.named_parameters():
    print(name, param.shape)
"""
0.weight torch.Size([13, 1])
0.bias torch.Size([13])
2.weight torch.Size([1, 13])
2.bias torch.Size([1])
"""    
# 0. 2. : module 순서 의미


###########################
####### OrderedDict #######
###########################

from collections import OrderedDict

# OrderedDict : named_layer
seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
]))

print(seq_model)
"""
Sequential(
  (hidden_linear): Linear(in_features=1, out_features=8, bias=True)
  (hidden_activation): Tanh()
  (output_linear): Linear(in_features=8, out_features=1, bias=True)
)
"""

for name, param in seq_model.named_parameters():
    print(name, param.shape)
"""
hidden_linear.weight torch.Size([8, 1])
hidden_linear.bias torch.Size([8])
output_linear.weight torch.Size([1, 8])
output_linear.bias torch.Size([1])
"""    
# name이 변경됨을 확인

print(seq_model.output_linear.bias)
"""
Parameter containing:
tensor([0.0096], requires_grad=True)
"""


##############################################
#### Monitoring gradients during training ####
##############################################
# Monitoring : 복잡하고 새로운 Neuron Network를 설계했을 때,
# 잘 돌아가는 아닌지, 알기어려우므로 모니터링을 통해 쉽게 파악 가능하다.

optimizer = optim.SGD(seq_model.parameters(), lr=1e-3) # lr 수정

training_loop(
    n_epochs = 5000, 
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_un_train,
    t_u_val = t_un_val, 
    t_c_train = t_c_train,
    t_c_val = t_c_val)
    
print('output', seq_model(t_un_val))
print('answer', t_c_val)
print('hidden', seq_model.hidden_linear.weight.grad)
"""
Epoch 1, Training loss 199.0593, Validation loss 63.2521
Epoch 1000, Training loss 5.0706, Validation loss 2.7776
Epoch 2000, Training loss 5.0318, Validation loss 7.0651
Epoch 3000, Training loss 2.9085, Validation loss 4.4777
Epoch 4000, Training loss 2.0808, Validation loss 3.4919
Epoch 5000, Training loss 1.9238, Validation loss 3.2763
output tensor([[12.4106],
        [-1.8676]], grad_fn=<AddmmBackward0>)
answer tensor([[11.],
        [-4.]])
hidden tensor([[ 0.0634],
        [ 0.0101],
        [ 0.0587],
        [-0.0044],
        [ 0.0504],
        [ 0.0634],
        [ 0.0796],
        [-0.0107]])
"""


#######################################
#### Comparing to the linear model ####
#######################################

from matplotlib import pyplot as plt

t_range = torch.arange(20., 90.).unsqueeze(1)   # 하나의 행렬로 제작

fig = plt.figure(dpi=600)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
# 'o' : 점
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
# 0.1 * t_range : normalizing
# .detach() : g 계산 없애기
# 'c-' : 선형
plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
# 'kx' : x
plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')
plt.savefig("comparing_to_the_linear_model.png", format="png")
# train_set에 대한 loss는 감소했다.
# validation_set은 ...
