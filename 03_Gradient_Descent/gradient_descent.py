import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_processor import load_data

#-------------------------------------------------#
#                 MAIN SOURCE CODE                #
#-------------------------------------------------#

x_train,y_train = load_data()
