import numpy as np
import matplotlib.pyplot as plt


def ReLU(x):
    ans = []
    for _x in x:
        if _x < 0:
            ans.append(0)
        else:
            ans.append(_x)
    return ans


def ELU(x, alpha=1):
    ans = []
    for _x in x:
        if _x < 0:
            ans.append(_x)
        else:
            ans.append(alpha*(np.exp(_x) - 1))
    return ans

x = np.arange(-5,2,0.01)

y1 = ReLU(x)
y2 = ELU(x)

plt.plot(x,y1)
plt.plot(x,y2)
plt.grid()
plt.legend("ReLU", "ELU")
plt.show()
