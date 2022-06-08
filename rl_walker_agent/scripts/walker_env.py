#!/usr/bin/env python3
'''
    By Shivam Chavan <shivam31199@gmail.com>
    Visit our website at www.melodic.pythonanywhere.com
'''

import gym
from std_msgs.msg import Float32
import rospy
import numpy as np
import time
from gym import utils, spaces
from geometry_msgs.msg import Pose
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection
from joint_publisher import JointPub
from walker_state import WalkerState
from controllers_connection import ControllersConnection


#register the training environment in the gym as an available one
reg = register(
    id='walker-v0',
    entry_point='walker_env:WalkerEnv')


class WalkerEnv(gym.Env):

    def __init__(self):
        
        # We assume that a ROS node has already been created
        # before initialising the environment
        self.reward = 0
        # gets training parameters from param server
        self.desired_pose = Pose()
        self.desired_pose.position.x = rospy.get_param("/desired_pose/x")
        self.desired_pose.position.y = rospy.get_param("/desired_pose/y")
        self.desired_pose.position.z = rospy.get_param("/desired_pose/z")

        self.running_step = rospy.get_param("/running_step")
        self.max_incl = rospy.get_param("/max_incl")
        self.max_height = rospy.get_param("/max_height")
        self.min_height = rospy.get_param("/min_height")
        
        self.joint_increment_value = rospy.get_param("/joint_increment_value")
        self.done_reward = rospy.get_param("/done_reward")
        self.alive_reward = rospy.get_param("/alive_reward")
        self.desired_force = rospy.get_param("/desired_force")
        self.desired_yaw = rospy.get_param("/desired_yaw")

        self.weight_r1 = rospy.get_param("/weight_r1")
        self.weight_r2 = rospy.get_param("/weight_r2")
        self.weight_r3 = rospy.get_param("/weight_r3")
        self.weight_r4 = rospy.get_param("/weight_r4")
        self.weight_r5 = rospy.get_param("/weight_r5")

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()
        self.reward_publisher = rospy.Publisher('/reward_per_step', Float32, queue_size=10)

        self.controllers_object = ControllersConnection(namespace=None)

        self.walker_state_object = WalkerState(   max_height=self.max_height,
                                                    min_height=self.min_height,
                                                    abs_max_roll=self.max_incl,
                                                    abs_max_pitch=self.max_incl,
                                                    joint_increment_value=self.joint_increment_value,
                                                    done_reward=self.done_reward,
                                                    alive_reward=self.alive_reward,
                                                    desired_force=self.desired_force,
                                                    desired_yaw=self.desired_yaw,
                                                    weight_r1=self.weight_r1,
                                                    weight_r2=self.weight_r2,
                                                    weight_r3=self.weight_r3,
                                                    weight_r4=self.weight_r4,
                                                    weight_r5=self.weight_r5
                                                )

        self.walker_state_object.set_desired_world_point(self.desired_pose.position.x,
                                                          self.desired_pose.position.y,
                                                          self.desired_pose.position.z)

        self.monoped_joint_pubisher_object = JointPub()
        

        '''
        - bum_zlj
        - bum_xlj
        - bum_ylj
        - knee_left
        - foot_lj
        - bum_zrj
        - bum_xrj
        - bum_yrj
        - knee_right
        - foot_rj

        '''

        ###################################################
        ############### Action Vals #######################

        low_action_joint_vals = np.array(
            [
                -0.340, -0.340, 0.0, -1.25, \
                -0.350, -0.340, -0.95, 0.0, \
                -1.25, 0.35,
            ],
            dtype=np.float32,
        )


        high_action_joint_vals = np.array(
            [
                0.340, 1.0, 1.0, 0.0, 0.65, \
                0.340, 0.340, 0.95, 0, 0.65, \
            ],
            dtype=np.float32,
        )

        ###################################################
        ############### obs Vals #########################
        

        obs_low_vals = np.array(
            [
                # body_pose
                -np.inf, -np.inf, -np.inf, \
                # body vel 
                -np.inf, -np.inf, -np.inf, \
                # body orientation
                -np.pi, -np.pi, -np.pi, \
                # body angular_vel
                -np.inf, -np.inf, -np.inf, \
                ########## joint states ##########
                -0.340, -0.340, 0.0, -1.25, -0.350, \
                -0.340, -0.95, 0.0,-1.25, 0.35, \
                ############# Angular Vel ########
                -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, \
                -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, \
                # Distance from desidre point 
                -np.inf, \
                ###########  contact forces ######
                0, 0, \
                ##################################
                ########## Previous Actions ######
                -0.340, -0.340, 0.0, -1.25, -0.350, \
                -0.340, -0.95, 0.0, -1.25, 0.35, \
                ################################
                

            ],
            dtype=np.float32,
        )


        obs_high_vals = np.array(
            [
                # body_pose
                np.inf, np.inf, np.inf, \
                # body vel 
                np.inf, np.inf, np.inf, \
                # body orientation
                np.pi, np.pi, np.pi, \
                # body angular_vel
                np.inf, np.inf, np.inf, \
                ########## joint states ##########
                0.340, 1.0, 1.0, 0.0, 0.65, \
                0.340, 0.340, 0.95, 0, 0.65, \
                ############# Angular Vel ########
                np.inf, np.inf, np.inf, np.inf, np.inf, \
                np.inf, np.inf, np.inf, np.inf, np.inf, \
                # Distance from desidre point 
                np.inf, \
                ###########  contact forces ######
                70, 70, \
                ##################################
                ########## Previous Actions ######
                0.340, 1.0, 1.0, 0.0, 0.65, \
                0.340, 0.340, 0.95, 0, 0.65, \
                ################################
            ],
            dtype=np.float32,
        )


        """
        For this version, we consider 10 actions
        1-2) Increment/Decrement of every joint
        """
        self.action_space = spaces.Box(low=low_action_joint_vals, high=high_action_joint_vals, dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.observation_space = spaces.Box(low=obs_low_vals, high=obs_high_vals, dtype=np.float32)

        a = np.array((self.desired_pose.position.x, self.desired_pose.position.y, self.desired_pose.position.z))
        b = np.array((0, 0, 0))

        self.dis = np.linalg.norm(a-b)

        self.previous_obs = np.array([
            0, 0, 0.78, 
            0, 0, 0, 
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            self.dis, 0, 0,
            0, 0, 0, 0, 0,            
            0, 0, 0, 0, 0,            
        ], dtype=np.float32)

        self.previous_action_commands = np.array([
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ], np.float32)

        self._seed()

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    # Resets the state of the environment and returns an initial observation.
    def reset(self):
        self.gazebo.pauseSim()
        self.gazebo.resetSim()
        self.gazebo.change_gravity(0.0, 0.0, 0.0)
        self.controllers_object.reset_monoped_joint_controllers()
        self.monoped_joint_pubisher_object.set_init_pose()
        self.walker_state_object.check_all_systems_ready()
        observation = self.walker_state_object.get_observations()
        self.gazebo.change_gravity(0.0, 0.0, -9.81)
        self.gazebo.pauseSim()
        state = np.array(observation, dtype=np.float32)
        print('Last Eps Total Reward: {}'.format(self.reward))
        return state

    def step(self, action):
        
        next_action_position = self.walker_state_object.get_action_to_position(action, self.previous_obs)
        self.gazebo.unpauseSim()
        self.monoped_joint_pubisher_object.move_joints(next_action_position)

        time.sleep(self.running_step)
        self.gazebo.pauseSim()

        observation = self.walker_state_object.get_observations()
        # self.previous_obs = self.walker_state_object.get_observations()
        reward,done = self.walker_state_object.process_data()
        msg = Float32()
        msg.data = reward
        self.reward_publisher.publish(msg)
        self.reward += reward
        # Get the State Discrete Stringuified version of the observations
        state = np.array(observation, dtype=np.float32)
        self.previous_obs = np.array(observation, dtype=np.float32)
        # print(state)
        return state, reward, done, {}