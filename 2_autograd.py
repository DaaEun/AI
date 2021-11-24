# autograd : Backpropagation all things / gradient 자동계산 툴

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

# requires_grad=True 
# - gradient를 계산할 것이다.
# - backward pass를 위해 forward pass 과정을 저장한다.(메모리 공간 확보)
params = torch.tensor([1.0, 0.0], requires_grad=True)
print(params.grad is None) # True 체크 


##################################
#### Using the grad attribute ####
##################################

# loss 계산
# t_p = model(t_u, *params)
loss = loss_fn(model(t_u, *params), t_c)
loss.backward() # backward 함수 호출

params.grad # params에 대한 미분 계산
# print(params.grad)
"""
tensor([4517.2969,   82.6000])
"""

# WARNING 
# 새로운 batch 단위 미분계산 적용을 위해서 grad=0 초기화 진행
if params.grad is not None:
    params.grad.zero_()


###########################
###### Training loop ######
###########################

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_()     #backward pass 과정전에 언제든 초기화 실행가능

        t_p = model(t_u, *params)   # forward
        loss = loss_fn(t_p, t_c)    # forward
        loss.backward()    

        with torch.no_grad():       # grad와 관계없는 과정 
            # params 업데이트는 grad에 영향 미치지 않아, 중간계산과정 저장을 하지 않는다.
            # 공간과 시간 절약
            # params -= lr * g
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params


##############################
########## Training ##########
##############################

training_params = training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0], requires_grad=True),
    t_u = t_un,     # Normalizing
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