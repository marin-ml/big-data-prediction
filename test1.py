
import numpy as np


a = [[1, 2, 3, 4, 5],
     [3, 2, 2, 5, 2]]

x = np.array(20).reshape((4, 5))

np.savetxt('test1.txt', x)