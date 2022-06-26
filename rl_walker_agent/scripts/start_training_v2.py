#!/usr/bin/python3

# import torch
import gym
from stable_baselines3 import DDPG, PPO, A2C, TD3, SAC
# from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import walker_env
import rospy
import rospkg

rospack = rospkg.RosPack()

if __name__ == "__main__":
    rospy.init_node('bipedal_gym', anonymous=True, log_level=rospy.INFO)
    env_name = 'walker-v0'
    env = gym.make(env_name)
    # print(env.action_space)
    # print(env.observation_space)
    k = check_env(env, warn= True, skip_render_check=True)
    print(k)
    pkg_path = rospack.get_path('rl_walker_agent')
    log_path =  pkg_path+'/training_results'
    # indir = pkg_path + '/training_results/500k_reward_eq_v5_effortController_step_a2c.zip'
    outdir_2 = pkg_path + '/training_results/100k_reward_eq_v5_newNetArch_effortController_step_td3'

    # env_main = DummyVecEnv([lambda: env])

    # new_arch  = [dict(pi=[512,256,256,256], vf=[512,256,256,256])]
    policy_kwargs = dict(net_arch=dict(pi=[512,256,256,256], qf=[512,256,256,256]))

    model = TD3('MlpPolicy', env, verbose=1, learning_rate=0.01, tensorboard_log=log_path, policy_kwargs=policy_kwargs)
    # model = A2C.load(indir, env= env_main,)
    model.learn(total_timesteps= 100000, log_interval=1000)

    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('rl_walker_agent')
    model.save(outdir_2)

