import math
from matplotlib import pyplot as plt

eps_start = 1.0
eps_end = 0.01
eps_decay = 100000

epsilon_by_frame = lambda frame_idx: eps_end + (eps_start - eps_end) * math.exp(-1. * frame_idx / eps_decay)
plt.plot([epsilon_by_frame(i) for i in range(300000)])
plt.title('Epsilon value over episodes')
plt.ylabel('Epsilon value')
plt.xlabel('Timesteps')
plt.show()