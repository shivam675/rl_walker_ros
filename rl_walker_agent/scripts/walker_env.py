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
        
        # self.joint_increment_value = rospy.get_param("/joint_increment_value")
        self.done_reward = rospy.get_param("/done_reward")
        # self.alive_reward = rospy.get_param("/alive_reward")
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
                                                    # joint_increment_value=self.joint_increment_value,
                                                    done_reward=self.done_reward,
                                                    # alive_reward=self.alive_reward,
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

        self.walker_joint_pubisher_object = JointPub()
        

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

        # low_action_joint_vals = np.array(
        #     [
        #         -10, -10, -10, -10, \
        #         -10, -10, -10, -10, \
        #         -10, -10,
        #     ],
        #     dtype=np.float32,
        # )


        # high_action_joint_vals = np.array(
        #     [
        #         10, 10, 10, 10, \
        #         10, 10, 10, 10, \
        #         10, 10,
        #     ],
        #     dtype=np.float32,
        # )

        ###################################################
        ############### obs Vals #########################
        

        obs_low_vals = np.array(
            [
                # body_pose
                # -np.inf, x val is omited
                -np.inf, -10,\
                # body vel 
                -4, -4, -4,\
                # body orientation
                -np.pi, -np.pi, -np.pi,\
                # body angular_vel
                -3, -3, -3,\
                ########## joint states ##########
                -2, -2, -2, -2, -2, \
                -2, -2, -2, -2, -2, \
                ############# Angular Vel ########
                -5, -5, -5, -5, -5,\
                -5, -5, -5, -5, -5,\
                # Distance from desidre point 
                # -np.inf,\
                ###########  contact forces ######
                # 0, 0,\
                ##################################
                ########## Previous Actions ######
                -20, -20, -20, -20, -20,\
                -20, -20, -20, -20, -20,\
                ################################
                

            ],
            dtype=np.float32,
        )


        obs_high_vals = np.array(
            [
                # body_pose
                # np.inf, x val is omited 
                np.inf, -10, \
                # body vel 
                4, 4, 4, \
                # body orientation
                np.pi, np.pi, np.pi, \
                # body angular_vel
                3, 3, 3, \
                ########## joint states ##########
                2, 2, 2, 2, 2, \
                2, 2, 2, 2, 2, \
                
                ############# Angular Vel ########
                5, 5, 5, 5, 5,\
                5, 5, 5, 5, 5,\
                # Distance from desidre point 
                # np.inf, \
                ###########  contact forces ######
                # 100, 100, \
                ##################################
                ########## Previous Actions ######
                20, 20, 20, 20, 20,\
                20, 20, 20, 20, 20,\
                ################################
            ],
            dtype=np.float32,
        )


        """
        For this version, we consider 10 actions with box space from -20 to + 20 
        1-2) Increment/Decrement of every joint
        """
        # self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,),dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,),dtype=np.float32)
        # self.action_space = spaces.MultiDiscrete(np.array([80, 80, 80, 80, 80, 80, 80, 80, 80, 80]))
        print(self.action_space)
        print(self.observation_space)
        self.reward_range = (-np.inf, np.inf)
        # self.observation_space = spaces.Box(low=obs_low_vals, high=obs_high_vals, dtype=np.float32)
        self.observation_space = spaces.Box(low=-20, high=20, shape=(41,),dtype=np.float32)

        
        self.previous_obs = [
            0, 0.78, 
            0, 0, 0, 
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,            
            0, 0, 0, 0, 0,            
        ]

        self.previous_action_commands = np.array([
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ], np.float32)
        # print(len(self.observation_space.sample()))
        self.step_number = 0

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
        self.controllers_object.reset_walker_joint_controllers()
        self.walker_joint_pubisher_object.set_init_pose()
        self.walker_state_object.check_all_systems_ready()
        observation = self.walker_state_object.get_observations()
        # print(len(observation))
        self.gazebo.change_gravity(0.0, 0.0, -9.81)
        self.gazebo.pauseSim()

        state = np.array(observation, dtype=np.float32)
        # print(len(state))
        # print('Last Eps Total Reward: {}'.format(self.reward), flush=True, end='\r')
        self.reward = 0
        self.step_number = 0
        self.previous_obs = [
            0, 0.78, 
            0, 0, 0, 
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,            
            0, 0, 0, 0, 0,            
        ]

        return state

    def step(self, action):
        self.step_number += 1
        action_torques = self.walker_state_object.dump_previous_actions(action, self.previous_action_commands ,self.step_number)
        self.previous_action_commands = action_torques
        self.gazebo.unpauseSim()
        self.walker_joint_pubisher_object.move_joints(action_torques)

        time.sleep(self.running_step)
        self.gazebo.pauseSim()

        observation = self.walker_state_object.get_observations()
        reward, done = self.walker_state_object.process_data()
        msg = Float32()
        msg.data = reward
        self.reward_publisher.publish(msg)
        # Get the State Discrete Stringuified version of the observations
        state = np.array(observation, dtype=np.float32)

        return state, reward, done, {}