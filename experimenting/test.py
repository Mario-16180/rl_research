import matplotlib.pyplot as plt
import numpy as np

# Generate two sets of 10000 data points, similar to loss values. One which is more or less stable and the other one with a lot of fluctuations. Then plot them both in the same figure.
# Generate data
x = np.arange(10000)
y1 = np.random.randn(10000) + 5
y2 = np.random.randn(10000) + 5
y2[2000:3000] = y2[2000:3000] + 10
y2[5000:6000] = y2[5000:6000] + 10
y2[7000:8000] = y2[7000:8000] + 10
y2[9000:10000] = y2[9000:10000] + 10
plt.plot(x, y1, label='stable')
plt.plot(x, y2, label='fluctuating')
plt.legend()
plt.show()
