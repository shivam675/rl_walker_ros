#!/usr/bin/python3
import gym
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import walker_env
import rospy
import rospkg


if __name__ == "__main__":
    rospy.init_node('bipedal_gym', anonymous=True, log_level=rospy.INFO)
    env_name = 'walker-v0'
    env = gym.make(env_name)
    print(env.action_space)
    print(env.observation_space)

    # episodes = 15
    # for episode in range(1,episodes+1):
    #     state = env.reset()
    #     done = False
    #     score = 0

    #     while not done:
    #         action = env.action_space.sample()
    #         n_state, reward, done, info = env.step(action=action)
    #         score += reward

    #     print('Episode: {}  Score: {} '.format(episode, score))
    # env.close()
    log_path = '/home/ros/custom_ai/src/walker/rl_walker_agent/training_results'
    # env_main = DummyVecEnv([lambda: env])
    model = DDPG('MlpPolicy', env, verbose=2, learning_rate=0.001, tensorboard_log=log_path)
    model.learn(total_timesteps= 300000)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('rl_walker_agent')
    outdir = pkg_path + '/training_results/300k_reward_eq_v2_step_ddpg'
    model.save(outdir)

