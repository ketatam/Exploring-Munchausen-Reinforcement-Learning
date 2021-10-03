import gym
from stable_baselines3 import SAC
from m_sac import MunchausenSAC

from make_env import make_env
import time

do_train = True
is_SAC = False
num_obst = 5
L0s = [-1.0, -2.0, -3.0]
MUNCHAUSEN_COEFS = [0.9, 0.5, 0.1, 0.05]

env = make_env("simple")

for l0 in L0s:
    for m in MUNCHAUSEN_COEFS:
        model = MunchausenSAC('MlpPolicy', env, verbose=0, l0=l0, munchausen_coef=m)
        print(f'Experiment with l0={l0}, and munchausen_coef{m}')
        model.learn(total_timesteps=500000)
        path = "Munchausen_SAC_l0_"+str(abs(int(l0)))+"_m_"+str(int(m*10))
        model.save(path)

        obs = env.reset()
        i = 0
        nb_success = 0
        counter = 0
        average_timesteps = 0
        while i<10000:
            counter += 1
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            #env.render()

            if done != 0 or counter>=500:
                i += 1
                if done == 1:
                    nb_success += 1
                    average_timesteps = average_timesteps + (counter - average_timesteps) / nb_success
                obs = env.reset()
                counter = 0
        print(f'{nb_success} successes out of {i} trials. Average time steps is: {average_timesteps}')
        print("------------------")
