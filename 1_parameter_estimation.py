# Converting Celsius to Fahrenheit
# 우리가 알고있는 화씨로 섭씨 구하는 공식을 이용하지 않기
# 오직, 데이터로만 learning 하기

import numpy as np
import torch

# torch.set_printoptions : 출력 인쇄 옵션
# edgeitems : 각 차원의 시작과 끝에서 요약된 배열 항목 수 (default = 3)
# precision : 부동 소수점 출력의 정밀도 자릿수 (default = 4)
# linewidth : 줄바꿈 삽입을 위한 줄당 문자 수 (default = 80)
torch.set_printoptions(edgeitems=2, precision=2, linewidth=75)  

# data 넣기
# t_c : 섭씨(Celsius) 데이터 / target
# t_u : 화씨(Fahrenheit) 데이터 / unkown data
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


############################################
## Choosing a linear model as a first try ##
############################################

# model 클래스
# x : input / t_u
# b : bias
# y = w*x + b
def model(t_u, w, b):
    return w*t_u + b    

# loss fucntion 클래스 
def loss_fn(t_p, t_c):
    # squared_diffs : (output - target)^2
    squared_diffs = (t_p - t_c)**2
    # MSE
    # mean() : 평균 구하기
    return squared_diffs.mean()

w = torch.ones(())
b = torch.zeros(())

# t_p : output 
t_p = model(t_u, w, b)
# print(t_p)
"""
tensor([35.70, 55.90, 58.20, 81.90, 56.30, 48.90, 33.90, 21.80, 48.40,
        60.40, 68.40])
"""

loss = loss_fn(t_p, t_c)
# print(loss)
"""
tensor(1763.88)
"""

#########################
#### Decreasing loss ####
#########################

# delta : w 변화량
delta = 0.1 

# loss gradient 계산
# gradient = loss 변화량 / w 변화량
loss_rate_of_change_w = \
    (loss_fn(model(t_u, w + delta, b), t_c) - loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)

# bias gradient 계산
loss_rate_of_change_b = \
    (loss_fn(model(t_u, w, b + delta), t_c) - loss_fn(model(t_u, w, b + delta), t_c)) / (2.0 * delta)    

# lr 지정
# 1e-2 = 0.01
learning_rate = 1e-2    

# w 업데이트
w = w - learning_rate * loss_rate_of_change_w

# b 업데이트
b = b - learning_rate * loss_rate_of_change_b

###########################################
## Applying the derivatives to the model ##
###########################################

# loss 미분
def dloss_fn(t_p, t_c):
    # size(0) : 행의 길이
    dsqm_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsqm_diffs

# model 변화량 / w 변화량
def dmodel_dw(t_u, w, b):
    return t_u  # array

# model 변화량 / b 변화량
def dmodel_db(t_u, w, b):
    return 1.0  # scala  

########################################
#### Defining The Gradient Function ####
########################################

# gradient 계산 통합
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    # batch 단위 processing
    # broadcasting rule 적용
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

####################################
#### Iterating to fit the model ####
####################################

# params : [w, b]
def training_loop(n_epochs, learning_rate, params, t_u, t_c, print_params = True):
    for epoch in range(1, n_epochs + 1):
        w, b = params

        t_p = model(t_u, w, b)                  # Forward pass 
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)     # Backward pass

        params = params - learning_rate * grad

        # print('Epoch %d, Loss %f' % (epoch, float(loss)))

        # 상세한 결과값 출력
        if epoch in {1, 2, 3, 10, 11, 99, 100, 4000, 5000}:  
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            if print_params:
                print('    Params:', params)
                print('    Grad:  ', grad)
        if epoch in {4, 12, 101}:
            print('...')

        # isfinite(input) : input이 유한하면 True
        # loss 값이 무한히 커지면
        if not torch.isfinite(loss).all():
            break 
            
    return params


# learning_1
training_params =  training_loop(
    n_epochs = 100, 
    learning_rate = 1e-2,   # 0.01
    params = torch.tensor([1.0, 0.0]),  # [w, b] 
    t_u = t_u, 
    t_c = t_c)  
print(training_params)    
"""
Epoch 1, Loss 1763.884766
    Params: tensor([-44.17,  -0.83])
    Grad:   tensor([4517.30,   82.60])
Epoch 2, Loss 5802484.500000
    Params: tensor([2568.40,   45.16])
    Grad:   tensor([-261257.41,   -4598.97])
Epoch 3, Loss 19408029696.000000
    Params: tensor([-148527.73,   -2616.39])
    Grad:   tensor([15109614.00,   266155.69])
...
Epoch 10, Loss 90901105189019073810297959556841472.000000
    Params: tensor([3.21e+17, 5.66e+15])
    Grad:   tensor([-3.27e+19, -5.76e+17])
Epoch 11, Loss inf
    Params: tensor([-1.86e+19, -3.27e+17])
    Grad:   tensor([1.89e+21, 3.33e+19])
tensor([-1.86e+19, -3.27e+17])
"""
# loss가 급격하게 증가하고, 결국 epoch 11번째에서 inf 되어 학습 중지
# 코드를 수정해보자.   

###############################
#### Hyperparameter tuning ####
###############################

# learning_2 : lr 줄이기
training_params =  training_loop(
    n_epochs = 100, 
    learning_rate = 1e-4,   # 0.0001
    params = torch.tensor([1.0, 0.0]),  
    t_u = t_u, 
    t_c = t_c)  
print(training_params)  
"""
Epoch 1, Loss 1763.884766
    Params: tensor([ 0.55, -0.01])
    Grad:   tensor([4517.30,   82.60])
Epoch 2, Loss 323.090515
    Params: tensor([ 0.36, -0.01])
    Grad:   tensor([1859.55,   35.78])
Epoch 3, Loss 78.929634
    Params: tensor([ 0.29, -0.01])
    Grad:   tensor([765.47,  16.51])
...
Epoch 10, Loss 29.105247
    Params: tensor([ 0.23, -0.02])
    Grad:   tensor([1.48, 3.05])
Epoch 11, Loss 29.104168
    Params: tensor([ 0.23, -0.02])
    Grad:   tensor([0.58, 3.04])
...
Epoch 99, Loss 29.023582
    Params: tensor([ 0.23, -0.04])
    Grad:   tensor([-0.05,  3.02])
Epoch 100, Loss 29.022667
    Params: tensor([ 0.23, -0.04])
    Grad:   tensor([-0.05,  3.02])
tensor([ 0.23, -0.04])
"""
# 조금씩 loss가 감소하는 것을 확인할 수 있다.
# 문제는 t_u, t_c 데이터의 스케일이 다르다는 점이다.
 
############################
#### Normalizing inputs ####
############################

# learning_3 : 두 input data 스케일을 비슷하여 적용 (= Normalizing)
t_un = 0.1 * t_u    # 정규화계산을 하여 적용하는 것이 원칙이나, 심플하게 해보자
training_params =  training_loop(
    n_epochs = 100, 
    learning_rate = 1e-2,   # 0.01
    params = torch.tensor([1.0, 0.0]),  
    t_u = t_un,     # Normalizing
    t_c = t_c)  
print(training_params)  
"""
Epoch 1, Loss 80.364342
    Params: tensor([1.78, 0.11])
    Grad:   tensor([-77.61, -10.64])
Epoch 2, Loss 37.574913
    Params: tensor([2.08, 0.13])
    Grad:   tensor([-30.86,  -2.39])
Epoch 3, Loss 30.871077
    Params: tensor([2.21, 0.12])
    Grad:   tensor([-12.46,   0.86])
...
Epoch 10, Loss 29.030489
    Params: tensor([ 2.32, -0.07])
    Grad:   tensor([-0.54,  2.93])
Epoch 11, Loss 28.941877
    Params: tensor([ 2.33, -0.10])
    Grad:   tensor([-0.52,  2.93])
...
Epoch 99, Loss 22.214186
    Params: tensor([ 2.75, -2.49])
    Grad:   tensor([-0.45,  2.52])
Epoch 100, Loss 22.148710
    Params: tensor([ 2.76, -2.52])
    Grad:   tensor([-0.44,  2.52])
tensor([ 2.76, -2.52])
""" 


# learning_4 : Normalizing 적용 + 학습횟수 증가
params =  training_loop(
    n_epochs = 5000,        # 증가
    learning_rate = 1e-2,   # 0.01
    params = torch.tensor([1.0, 0.0]),  
    t_u = t_un,     # Normalizing
    t_c = t_c,
    print_params = False)  
print(params) 
"""
Epoch 1, Loss 80.364342
Epoch 2, Loss 37.574913
Epoch 3, Loss 30.871077
...
Epoch 10, Loss 29.030489
Epoch 11, Loss 28.941877
...
Epoch 99, Loss 22.214186
Epoch 100, Loss 22.148710
...
Epoch 4000, Loss 2.927680
Epoch 5000, Loss 2.927648
tensor([  5.37, -17.30])
"""
# 실제 value 값 - w = 5.5556 / b = -17.7778
# 실제값과 학습값이 유사해졌다.
# 완전히 일치하지 않는 이유는 데이터 자체가 측정 오차값이 존재하기 때문이다.

#################################
########## Visualizing ##########
#################################

from matplotlib import pyplot as plt    # 그림 그리기 

# argument unpacking
# *params = params[0], params[1]
t_p = model(t_un, *params)

# 그림 초기화
fig = plt.figure(dpi = 600)   

plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")

# detach() : pytorch에서 기본적으로 미분계산을 위한 정보를 저장하는데, 이를 False
# numpy() : array로 입력
plt.plot(t_u.numpy(), t_p.detach().numpy())     # 직선
plt.plot(t_u.numpy(), t_c.numpy(), 'o')         # 점
plt.savefig("temp_unknown_plot.png", format="png")  # bookskip
