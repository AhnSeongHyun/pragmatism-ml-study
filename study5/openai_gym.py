import gym
from gym import spaces
from gym import envs
from gym import wrappers

print(envs.registry.all())
space = spaces.Discrete(8)
x = space.sample()
assert space.contains(x)
assert space.n
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './caripole-experiment-1')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

for i_episode in range(20):
    observation = env.reset() # start
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished afer {} timestpeps".format(t+1))
            break




