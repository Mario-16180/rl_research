# Read the csv file and print the data
import pandas as pd

# Read the csv file
rewardbounds_per_env = pd.read_csv('experimenting/rl_utils/reward_data_per_environment.csv', delimiter=' ', header=0)
print(rewardbounds_per_env.iloc[0,1])