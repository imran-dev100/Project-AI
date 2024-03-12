import csv
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------#
#       MODEL FUNCTIONFOR LINEAR REGRESSION       #
#-------------------------------------------------#

def compute_model_output(x, w, b):
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
file = './data/ice-cream-sales-temperature.csv'


### Initializing empty arrays for training data
x_train = np.array([]) # Temperature in Fahrenheit
y_train = np.array([]) # Sales in percentage



### Parsing data set from file
with open(file, mode ='r')as file:
  csvFile = csv.reader(file)
  header = next(csvFile) # skipping headers
  
  for line in csvFile:
    x_train = np.append(x_train, int(line[0]))
    y_train = np.append(y_train, float(line[1]))

m = len(x_train)



### Printing the data set
# for i in range(m):
#     print(f"x^{i} = {x_train[i]}, y^{i} = {y_train[i]}")



### Plotting a graph with x as marker and blue as color
plt.scatter(x_train, y_train, marker = 'x', c = '#0080FF')
plt.title('Ice cream sales with different temperature')
plt.xlabel('Temperature in Fahrenheit')
plt.ylabel('Sales in percentage')
plt.show()


# Assuming weight for the cost function
w = 24

# Assuming bias for the cost function
b = 24

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='#000080',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='#8080FF',label='Actual Values')

plt.title('Ice cream sales with different temperature')
plt.xlabel('Temperature in Fahrenheit')
plt.ylabel('Sales in percentage')
plt.legend()
plt.show()