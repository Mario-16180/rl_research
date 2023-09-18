import gym
import rware

if __name__ == '__main__':
    #env = gym.make("rware:rware-tiny-2ag-v1")
    #env = gym.make('rware:rware-tiny-2ag-v1')
    env = gym.make("rware-tiny-2ag-v1")
    #env = gym.make('CartPole-v0')
    obs = env.reset()  # a tuple of observations
    done = [False, False]
    while (not done[0]) & (not done[1]):
        actions = env.action_space.sample()  # the action space can be sampled
        print(actions)  # (1, 0)
        n_obs, reward, done, info = env.step(actions)
        #env.render()
        print(done)    # [False, False]
        print(reward)  # [0.0, 0.0]
    #env.close()