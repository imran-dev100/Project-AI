import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_processor import load_data, plot_data_graph
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------#
#      MODEL FUNCTION FOR LINEAR REGRESSION       #
#-------------------------------------------------#

def compute_linear_regression(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


#-------------------------------------------------#
#                 MAIN SOURCE CODE                #
#-------------------------------------------------#

x_train,y_train = load_data()
m = len(x_train)


### Printing the data set
# for i in range(m):
#     print(f"x^{i} = {x_train[i]}, y^{i} = {y_train[i]}")

plot_data_graph(x_train, y_train)




# Assuming weight for the cost function
w = 24

# Assuming bias for the cost function
b = 24

tmp_f_wb = compute_linear_regression(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='#000080',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='#8080FF',label='Actual Values')

plt.title('Ice cream sales with different temperature')
plt.xlabel('Temperature in Fahrenheit')
plt.ylabel('Sales in percentage')
plt.legend()
plt.show()