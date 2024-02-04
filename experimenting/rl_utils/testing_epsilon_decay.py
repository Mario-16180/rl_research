import math
from matplotlib import pyplot as plt

eps_start = 0.8 # Possible values [0.8, 0.85, 0.9, 0.95, 1.0]
eps_end = 0.05 # Possible values [0.0001, 0.001, 0.01, 0.05]
time_steps = 300_000

# for proportional_factor in [1/2, 1/3, 1/4, 1/5]: # The lower the proportional factor, the higher the decay

#     eps_decay = time_steps * proportional_factor

#     epsilon_by_frame = lambda frame_idx: eps_end + (eps_start - eps_end) * math.exp(-1. * frame_idx / eps_decay)
#     plt.plot([epsilon_by_frame(i) for i in range(time_steps)])
#     plt.title(f'Epsilon value over episodes, proportional factor = {proportional_factor}')
#     plt.ylabel('Epsilon value')
#     plt.xlabel('Timesteps')
#     plt.show()

eps_decay = 2e-5# The lower the value, the slower the decay [7.5e-3 to 2e-5]
# def select_action(self, env, state, epsilon_start, epsilon_decay, epsilon_min, current_step, device):
# eps_threshold = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-1. * current_step * epsilon_decay)
epsilon_by_frame = lambda frame_idx: eps_end + (eps_start - eps_end) * math.exp(-1. * frame_idx * eps_decay)
plt.plot([epsilon_by_frame(i) for i in range(time_steps)])
plt.title(f'Epsilon value over episodes, eps decay = {eps_decay}')
plt.ylabel('Epsilon value')
plt.xlabel('Timesteps')
plt.show()