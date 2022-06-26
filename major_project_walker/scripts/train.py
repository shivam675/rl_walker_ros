#!/usr/bin/python3

from stable_baselines3 import PPO, A2C, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from bipedal_env_v7 import walkerEnv7
from policies import CustomActorCriticPolicy
# import numpy as np
import rospkg
# import torch as th

lr_start = 100 * (1e-6)
lr_final = 1 * (1e-6)
lr_scale = 1




rospack = rospkg.RosPack()

if __name__ == "__main__":
    # env = walkerEnv7()
    # env = DummyVecEnv([lambda: env])
    env = make_vec_env(walkerEnv7, n_envs=32)
    pkg_path = rospack.get_path('major_project_walker')
    log_path =  pkg_path+'/training_results'
    # in_dir = pkg_path + '/training_results/1900K_ppo_model.zip'
    outdir_2 = pkg_path + '/training_results/{}m_ppo_model.zip'
    outdir_for_replaybuffer = pkg_path + '/training_results/{}m_td3_replay_buffer'
    # env_main = DummyVecEnv([lambda: env])
    # policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[700,300,300,300], vf=[700,300,300,300])])
    # policy_kwargs = dict(net_arch=[dict(pi=[800,600,400], vf=[600,400,400])])
    # policy_kwargs = dict(net_arch=dict(pi=[800,600,400], qf=[800,600,400]))


    # learning_rate_schedule = LinearDecay(lr_start, lr_final).value
    model = PPO('MlpPolicy', 
                env, 
                verbose=4, 
                learning_rate=0.0003, 
                # learning_starts=1000,
                tensorboard_log=log_path, 
                n_steps=2048,
                batch_size=64,
                gae_lambda=0.95,
                clip_range=0.18,
                # gradient_steps=1,
                # train_freq=1,
                n_epochs=10,
                ent_coef=0.0,
                gamma=0.999,
                # policy_kwargs=policy_kwargs
                )
    # model = PPO.load(in_dir, env=env)
    for i in range(1, 1000):
        model.learn(total_timesteps= 2000000, log_interval=1)
        model.save(outdir_2.format(i*2))
        # model.save_replay_buffer(outdir_for_replaybuffer.format(i))

