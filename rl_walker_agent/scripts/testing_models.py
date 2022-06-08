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

    model_path = '/home/ros/custom_ai/src/walker/rl_walker_agent/training_results/300k_reward_eq_v2_step_ddpg.zip'
    model = DDPG.load(model_path, env=env)
    # evaluate_policy(model, env=env, n_eval_episodes=10)

    episodes = 15
    for episode in range(1,episodes+1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action=action)
            score += reward

        print('Episode: {}  Score: {} '.format(episode, score))