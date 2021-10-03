import gym
from stable_baselines3 import A2C, DQN,SAC,TD3
from m_sac import MunchausenSAC

from make_env import make_env
import time

do_train = True
is_SAC = False
num_obst = 5

env = make_env("simple_no_fancy_init")
#env = gym.make('CartPole-v1')

if do_train:
    if is_SAC:
        model = SAC('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=100)
        model.save("SAC_bps")
    else:
        model = MunchausenSAC('MlpPolicy', env, verbose=1, munchausen_coef=0.5, l0=-2.0)
        model.learn(total_timesteps=2000000)
        model.save("Munchausen_SAC_bps_no_rel")
else:
    if num_obst==5:
        if is_SAC:
            model = SAC.load("SAC_5_obst", env=env)
        else:
            model = SAC.load("Munchausen_SAC_5_obst", env=env)

    if num_obst==10:
        if is_SAC:
            model = SAC.load("SAC_10_obst", env=env)
        else:
            model = SAC.load("Munchausen_SAC_10_obst", env=env)

obs = env.reset()
#for i in range(1000):
i = 0
nb_success = 0
counter = 0
average_timesteps = 0
while i<10000:
    counter += 1
    action, _state = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.05)
    #if i%100==0:
    if done != 0 or counter>=200:
        i += 1
        if done == 1:
            nb_success += 1
            average_timesteps = average_timesteps + (counter - average_timesteps) / nb_success
        time.sleep(2)
        obs = env.reset()
        counter = 0
        print(i)

print(f'{nb_success} successes out of {i} trials. Average time steps is: {average_timesteps}')
