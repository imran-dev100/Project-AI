import csv
import numpy as np
import matplotlib.pyplot as plt

file='data/ice-cream-sales-temperature.csv'

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

# for i in range(m):
#     print(f"x^{i} = {x_train[i]}, y^{i} = {y_train[i]}")

# Plotting a graph with x as marker and blue as color
plt.scatter(x_train, y_train, marker = 'x', c = 'blue')
plt.title('Ice cream sales with different temperature')
plt.xlabel('Temperature in Fahrenheit')
plt.ylabel('Sales in percentage')
plt.show()