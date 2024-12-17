# Check what happens when we increase the number of points between the two points

import numpy as np
import matplotlib.pyplot as plt
num_steps = 10
start = np.array([-5.0])
end = np.array([5.0])
samples = start + (end - start) * np.linspace(0, 1, num=num_steps)

# print(samples)


