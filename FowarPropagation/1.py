import numpy as np
alpha = 0.1
a = 1 / (1 + np.exp(2.3))
print(f"{a:.4f}")

b = -0.8*a*(1-a)*0.4 * alpha
print(f"{b:.5f}")

w11 = 2.0
w11 -= b
print(f"{w11:.5f}")
