import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
"""
Density map visualization
"""
load_npy = np.load(".npy")
np.set_printoptions(threshold=np.inf)
plt.imshow(load_npy, cmap=cm.jet)
plt.show()
