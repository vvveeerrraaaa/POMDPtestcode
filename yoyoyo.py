#np.savez practice
# Vera Resendez

import numpy as np

for x in range(5):
    danger = [1,2,3,4,5,6]*x
    ready = [2,4,5]*x
    go  = [6,7,8]*x
    np.savez('each'+str(x)+'.npz',danger,ready,go)

from numpy import load

data = load('1gru4.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])
