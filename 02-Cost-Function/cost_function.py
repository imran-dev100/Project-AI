import numpy as np
import matplotlib.pyplot as plt
import csv

#-------------------------------------------------#
#              COMPUTE COST FUNCTION              #
#-------------------------------------------------#

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost

file = './data/ice-cream-sales-temperature.csv'

x_train = np.array([]) # Temperature in Fahrenheit
y_train = np.array([]) # Sales in percentage

### Parsing data set from file
with open(file, mode ='r')as file:
  csvFile = csv.reader(file)
  header = next(csvFile) # skipping headers
  
  for line in csvFile:
    x_train = np.append(x_train, int(line[0]))
    y_train = np.append(y_train, float(line[1]))

# Plotting a graph with x as marker and blue as color
plt.scatter(x_train, y_train, marker = 'x', c = '#0080FF')
plt.title('Ice cream sales with different temperature')
plt.xlabel('Temperature in Fahrenheit')
plt.ylabel('Sales in percentage')
plt.show()



b = 24

w_array = np.arange(-24, 25, 1)  # The stop value is exclusive, so use 25 to include 24 with step 1
# w_array = np.linspace(-24, 24, 10)  # 10 values evenly spaced between -24 and 24.
# w_array = np.random.uniform(-24, 24, size=10)  # Array of 10 random floats between -24 and 24.
# w_array = np.random.randint(-24, 25, size=10)  # Array of 10 random integers between -24 and 24, inclusive.
# w_array = np.full((5,), -24)  # Creates an array of 5 elements, each initialized to -24.


cost_array = np.array([])

for w in w_array:
   cost = compute_cost(x_train, y_train, w, b)
   print(f"cost: {cost}")
   cost_array = np.append(cost_array, cost)


# Plotting a graph with x as marker and blue as color
plt.plot(w_array, cost_array, marker = 'x', c = '#8080FF')
plt.title('Cost function plot')
plt.xlabel('w')
plt.ylabel('cost')
plt.show()