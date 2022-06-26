#!/usr/bin/env python3

import time
import gym
from bipedal_env_v7 import walkerEnv7
import mujoco_py
# env = walkerEnv7()
import rospkg
from stable_baselines3 import SAC

env = gym.make("Humanoid-v3")

pkg = rospkg.RosPack()

ma = pkg.get_path('major_project_walker')
path = ma + '/training_results/sac.zip'
print(path)



steps = 1000
n_updates = 200

model = SAC.load(path=path)

for i in range(steps):
    state = env.reset()
    done = False
    for i in range(n_updates):
        env.render()
        action, _ = model.predict()
        # print(action-300)
        state, rew, done, info = env.step(action)
        print(len(state), len(env.observation_space.sample()), rew)
        # time.sleep(0.001)
        # print(state)
        # print(np.shape(state)) 
        # print(f'Checking if the state is part of the observation space: {env.observation_space.contains(state)}')
        # state
        # time.sleep(0.005)
        if done:
            break
    # break
    # viewer.loop_once()
    # data, width, height = viewer.get_image()
    # img = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
    # imsave('imgs/out_' + str(i) + '.png', img)
    # for j in range(skip):
        # model.step()
# print()
end = time.time()
# print(end - start)

# viewer.finish()
viewer = None