import numpy as np
import math
import matplotlib.pyplot as plt

def zigzag_plot(num_trianges = 3, min_val = 0, max_val = 1, triangle_base = 2):
    x = np.linspace(0, num_trianges * triangle_base, num=1000)
    tan_val = 2*max_val/triangle_base
    print(tan_val)
    y = [i*tan_val for i in x]
    y = [abs(i - triangle_base * np.floor((i+triangle_base/2)/triangle_base))* tan_val for i in x]
    plt.plot(x,y)
