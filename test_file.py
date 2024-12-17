# Check what happens when we increase the number of points between the two points
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

num_steps = 1000
start = np.array([-5.0])
end = np.array([5.0])
samples = start + (end - start) * np.linspace(0, 1, num=num_steps)

# print(samples)
norm_dist = stats.norm(0, 1)
probabilities = norm_dist.pdf(samples)
normalized_probabilities = probabilities / np.max(probabilities)    

print(probabilities[num_steps//2])
print(normalized_probabilities[num_steps//2])

