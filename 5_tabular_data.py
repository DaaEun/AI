# Representing tabular data

import numpy as np
import torch
torch.set_printoptions(edgeitems=2, precision=2, linewidth=75)


####################################
#### Loading a wine data tensor ####
####################################

import csv
wine_path = "tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
print(wineq_numpy)
"""
[[ 7.    0.27  0.36 ...  0.45  8.8   6.  ]
 [ 6.3   0.3   0.34 ...  0.49  9.5   6.  ]
 [ 8.1   0.28  0.4  ...  0.44 10.1   6.  ]
 ...
 [ 6.5   0.24  0.19 ...  0.46  9.4   6.  ]
 [ 5.5   0.29  0.3  ...  0.38 12.8   7.  ]
 [ 6.    0.21  0.38 ...  0.32 11.8   6.  ]]
"""

col_list = next(csv.reader(open(wine_path), delimiter=';'))
print(wineq_numpy.shape)
print(col_list)
"""
(4898, 12) 
['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
"""

wineq = torch.from_numpy(wineq_numpy)
print(wineq.shape, wineq.dtype)
"""
torch.Size([4898, 12]) torch.float32
"""