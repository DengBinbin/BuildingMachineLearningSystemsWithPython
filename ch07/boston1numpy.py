# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# This script shows an example of simple (ordinary) linear regression

import numpy as np
from sklearn.datasets import load_boston
import pylab as plt


boston = load_boston()
#首先选择第五维的数据作为输入，此时的Residual为7.64
x = boston.data[:,5]
# x = np.array([[v] for v in x])

#在第五维的数据基础上，加上一个偏移项作为输入，此时的Residual为6.60
# x = np.array([[v, 1] for v in x])

#我们使用多维回归，使用np.concatenate将原始的输入v和1合并起来，此时的Residual为4.68
x = np.array([np.concatenate((v, [1])) for v in boston.data])
y = boston.target

# np.linal.lstsq implements least-squares linear regression
s, total_error, _, _ = np.linalg.lstsq(x, y, rcond=-1)

rmse = np.sqrt(total_error[0] / len(x))
print('Residual: {}'.format(rmse))

# Plot the prediction versus real:
plt.plot(np.dot(x, s), boston.target, 'ro')

# Plot a diagonal (for reference):
plt.plot([0, 50], [0, 50], 'g-')
plt.xlabel('predicted')
plt.ylabel('real')
plt.show()
