import copy
from stable_baselines3.common.vec_env import VecVideoRecorder
import gym
from stable_baselines3 import SAC
from m_sac import MunchausenSAC

from make_env import make_env
from copy import deepcopy
import time

env = make_env("simple_no_fancy_init")

SAC_model = SAC.load("trained_agents/SAC_bps_no_rel", env=env)
M_SAC_model = MunchausenSAC.load("trained_agents/Munchausen_SAC_bps_no_rel", env=env)


obs = env.reset()
m_obs = obs
m_env = deepcopy(env)

temp_env = deepcopy(env)
temp_obs = obs
m_temp_env = deepcopy(env)
m_temp_obs = obs


i = 0

nb_success = 0
counter = 0
average_timesteps = 0
done = 0

m_nb_success = 0
m_counter = 0
m_average_timesteps = 0
m_done = 0

common_successes = 0
is_deterministic = True
while i < 1000:
    while done == 0:
        counter += 1
        action, _state = SAC_model.predict(obs, deterministic=is_deterministic)
        obs, reward, done, info = env.step(action)

    while m_done == 0:
        m_counter += 1
        m_action, m_state = M_SAC_model.predict(m_obs, deterministic=is_deterministic)
        m_obs, m_reward, m_done, m_info = m_env.step(m_action)

    i += 1

    if done == 1:
        nb_success += 1
        average_timesteps = average_timesteps + (counter - average_timesteps) / nb_success

    if m_done == 1:
        m_nb_success += 1
        m_average_timesteps = m_average_timesteps + (m_counter - m_average_timesteps) / m_nb_success

    #"""
    if m_done == 1 and done == 2:
        temp_done = 0
        m_temp_done = 0

        env = VecVideoRecorder(env, 'GIFs/',
                               record_video_trigger=lambda x: x == 0, video_length=100,
                               name_prefix="SAC-{}".format(i))

        while temp_done == 0:
            temp_action, temp_state = SAC_model.predict(temp_obs, deterministic=is_deterministic)
            temp_obs, temp_reward, temp_done, temp_info = temp_env.step(temp_action)
            temp_env.render()
            #time.sleep(0.05)
        #time.sleep(2)
        temp_env.close()

        while m_temp_done == 0:
            m_temp_action, m_temp_state = M_SAC_model.predict(m_temp_obs, deterministic=is_deterministic)
            m_temp_obs, m_temp_reward, m_temp_done, m_temp_info = m_temp_env.step(m_temp_action)
            m_temp_env.render()
            time.sleep(0.05)
        time.sleep(2)
        break
        #m_temp_env.close()
    #"""


    if done==1 and m_done==1:
        common_successes += 1

    obs = env.reset()
    m_obs = obs
    m_env = deepcopy(env)

    temp_env = deepcopy(env)
    temp_obs = obs

    m_temp_env = deepcopy(env)
    m_temp_obs = obs

    counter = 0
    done = 0
    m_counter = 0
    m_done = 0


print(f'SAC: {nb_success} successes out of {i} trials. Average time steps is: {average_timesteps}')
print(f'M_SAC: {m_nb_success} successes out of {i} trials. Average time steps is: {m_average_timesteps}')
print(f'number of common successes: {common_successes}')