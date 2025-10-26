import torch
import matplotlib.pyplot as plt

# Generate 10,000 random samples from a standard normal distribution (mean=0, std=1)
x = torch.randn(10000)

# Plot histogram (density=True makes it show probability density instead of counts)
plt.hist(x, bins=50, density=True, alpha=0.6, label='Samples')

# Make a smooth curve for reference
# import numpy as np
# xs = np.linspace(-4, 4, 200)
# ys = 1/(np.sqrt(2*np.pi)) * np.exp(-xs**2 / 2)
# plt.plot(xs, ys, 'r', label='Ideal Gaussian')

# plt.title("Gaussian (Normal) Distribution")
# plt.xlabel("Value")
# plt.ylabel("Probability Density")
# plt.legend()
plt.show()
