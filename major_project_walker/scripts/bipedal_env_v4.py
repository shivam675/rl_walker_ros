#!/usr/bin/env python3

import mujoco_py
import numpy as np
from gym.envs.mujoco import mujoco_env
import gym
from gym import utils, spaces
import rospkg
from geometry_msgs.msg import Vector3
from mujoco_py.builder import MujocoException
import typing as typ
import tf
np.set_printoptions(suppress=True)

rospk = rospkg.RosPack()
pkg_path = rospk.get_path('major_project_walker')
model_path = pkg_path + "/model/walker3d.xml"





class walkerEnv4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, exclude_current_positions_from_observation=True):
        self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)
        self.model_path = model_path
        # frame_skip = 10
        super(walkerEnv4).__init__()
        self.model = mujoco_py.load_model_from_path(self.model_path)
        
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.action_space = spaces.MultiDiscrete([600,600,600,600,600,600,600,600])
        self.action_throttle = 1000
        # self.action_space = spaces.MultiDiscrete([2000,2000, 2000, 2000, 2000, 2000])
        # self.action_space = spaces.Discrete(8)
        self.observation_space= spaces.Box(high=np.inf, low= -np.inf, shape=(66,))
        
        self.frame_skip = 5
        self.ep_dur_max = 3000
        self.step_number = 0 
        self.deltaTime = 0.001*self.frame_skip
        ############ class attributes ###########
        self.healthy_z_range = (0.4, 1.5)
        self.abs_max_roll = 1.4
        self.abs_max_pitch = 1.4
        self.x_velocity, self.y_velocity, self.z_velocity =  0, 0, 0 
        self.all_joint_vels = [0, 0, 0, 0, 0, 0, 0, 0]
        self.body_rpy_velocities = np.array([0, 0, 0])
        # self.list_of_joints = ['right_hip', 'right_knee', 'left_hip', 'left_knee', 'left_ankle', 'right_ankle']

        ########################################################3

        self.init_qpos = np.array([0,  0, 1.08, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, ], dtype=np.float32)
        self.init_qvel = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,], dtype=np.float32)

        
        self.qpos_indices = [ "COM_POSX", "COM_POSY", "COM_POSZ",
                "TRUNK_ROT_X", "TRUNK_ROT_Y", "TRUNK_ROT_Z",
                "HIP_SAG_ANG_R", "HIP_FRONT_ANG_R",
                "KNEE_ANG_R", "ANKLE_ANG_R",
                "HIP_SAG_ANG_L", "HIP_FRONT_ANG_L",
                "KNEE_ANG_L", "ANKLE_ANG_L"]

        self.qvel_indices = ["COM_VELX", "COM_VELY", "COM_VELZ",
                "TRUNK_ANGVEL_X", "TRUNK_ANGVEL_Y", "TRUNK_ANGVEL_Z",
                "HIP_SAG_ANGVEL_R", "HIP_FRONT_ANGVEL_R",
                "KNEE_ANGVEL_R", "ANKLE_ANGVEL_R",
                "HIP_SAG_ANGVEL_L", "HIP_FRONT_ANGVEL_L",
                "KNEE_ANGVEL_L", "ANKLE_ANGVEL_L"]


        ############## set all weights ###########
        self.r6 = 0
        self.terminate_when_unhealthy = True
        self.healthy_reward_weight = 10
        self.forward_reward_weight = 8
        self.ctrl_cost_weight = 1
        self.z_pos_reward = 20
        self.orientation_cost_weight = 14
        self.reset_noise_scale = 0.001

    def get_COM_Z_position(self):
        return self.sim.data.qpos[self._get_COM_indices()[-1]]

    def mass_center(self, model, sim):
        mass = np.expand_dims(model.body_mass, 1)
        xpos = sim.data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0:3].copy()

    def get_base_rpy(self):
        euler_rpy = Vector3()
        euler_rpy.x = self.sim.data.qpos[self._get_trunk_rot_joint_indices()[0]]
        euler_rpy.y = self.sim.data.qpos[self._get_trunk_rot_joint_indices()[1]] 
        euler_rpy.z = self.sim.data.qpos[self._get_trunk_rot_joint_indices()[2]]
        return euler_rpy

    def walker_orientation_ok(self):
        orientation_rpy = self.get_base_rpy()
        roll_ok = self.abs_max_roll > abs(orientation_rpy.x)
        pitch_ok = self.abs_max_pitch > abs(orientation_rpy.y)
        orientation_ok = roll_ok and pitch_ok
        return orientation_ok
    
    def get_z_value_reward(self):
        # 0.34 is the robots height
        return min(((1.2 - self.get_COM_Z_position())*self.z_pos_reward), 0)

    
    def get_observations(self):
        data = self.sim.data
        position = data.qpos.flat.copy()
        # position = data.qfrc_actuator.flat.copy()
        angular_position = [self.sim.data.qpos[x] for x in self._get_actuated_joint_indices()]
        angular_velocities = self.all_joint_vels
        self.x_velocity = self.sim.data.qvel[0]  
        self.y_velocity = self.sim.data.qvel[1] 
        self.z_velocity = self.sim.data.qvel[2]

        # angular_orientation = self.get_base_rpy()
        return np.concatenate((
                                self.sim.data.qpos,
                                self.sim.data.qvel,
                                self.sim.data.qfrc_actuator,
                                np.array([self.x_velocity, self.y_velocity, self.z_velocity,
                                int(self.has_ground_contact()[0]), int(self.has_ground_contact()[1]),
                                self.body_rpy_velocities[0], self.body_rpy_velocities[1], self.body_rpy_velocities[2]]),
                                # np.array(angular_position),
                                angular_velocities,
                                self.previous_actions,
        ))

    ################################################
    ########## Properties ##########################

    ##################################################
    ############  Get reward functions ###############

    def actuator_control_cost(self):
        control_cost = self.ctrl_cost_weight * (np.sum(abs(self.sim.data.qfrc_actuator.flat.copy()))**2)
        return control_cost

    def get_reward(self):
        ############ Get obs reward ################
        orientation_rpy = self.get_base_rpy()
        orientation_cost = abs(orientation_rpy.x) + abs(orientation_rpy.y) + abs(orientation_rpy.z)
        ############################################
        # r1 = self.actuator_control_cost()
        r2 = self.forward_reward_weight * self.x_velocity
        r3 = 0.2 * self.healthy_reward_weight
        r4 = orientation_cost*self.orientation_cost_weight
        r5 = self.get_z_value_reward()

        t_total =  r2 + r3 + r5 -r4
        # print(t_total)
        return t_total

    def is_done(self):
        done_4 = False
        done_3 = False
        done_1 = False
        min_z, max_z = self.healthy_z_range
        
        done_1 = not (min_z < self.sim.data.qpos[2] < max_z)
        # done_2 = not self.walker_orientation_ok()
        # print(self.step_number, self.x_velocity)
        done_3 = self.step_number  > self.ep_dur_max

        # print(self.step_number)
        done_4 = self.sim.data.qpos[0] >= 6
            
        # print(done_1, done_3, done_4)
        # print(self.sim.data.qpos[2])
        return (done_1 or done_3), done_4
        
    ##################################################

    def step(self, action):
        #++++++++++++++++++++++#
        self.viewer.render()
        action = action - 300
        self.step_number += 1
        #++++++++++++++++++++++#
        self.previous_actions = action
        #=============================#
        # xyz_position_before = self.mass_center(self.model, self.sim)
        major_angles_before = self.sim.data.qpos[6:].copy()
        rpy_before = self.get_base_rpy()
        #--------------------------------------#
        try:
            # self.do_simulation(action, self._frame_skip)
            self.do_simulation(action, self.frame_skip)
            # self.render()
        # If a MuJoCo Exception is raised, catch it and reset the environment
        except MujocoException as mex:
            obs = self.reset()
            return obs, 0, True, {}
        #--------------------------------------#
        # xyz_position_after = self.mass_center(self.model, self.sim)
        major_angles_after = self.sim.data.qpos[6:].copy()
        rpy_after = self.get_base_rpy()
        ###############################################
        # xyz_velocity = (xyz_position_after - xyz_position_before) / (self.deltaTime)
        # self.x_velocity, self.y_velocity, self.z_velocity = xyz_velocity
        self.all_joint_vels = (major_angles_after - major_angles_before)/(self.deltaTime)
        self.body_rpy_velocities = np.array([rpy_after.x - rpy_before.x, rpy_after.y - rpy_before.y, rpy_after.z - rpy_before.z])/(self.deltaTime)

        reward = self.get_reward()
        obs = self.get_observations()
        # print(obs)
        done_fr, done_good = self.is_done()
        if done_fr:
            return obs, -500, done_fr, {}
        
        if done_good:
            return obs, reward+100, done_good, {}
        # print(self.has_ground_contact())
        return obs, reward, done_good, {}

    def reset_model(self):    
        self.step_number = 0
        self.r6 = 0
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale
        qpos = self.init_qpos + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel  + np.random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.previous_actions = np.array([0, 0, 0, 0, 0, 0, 0, 0,], dtype=np.float32)
        # self.true_actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.set_state(qpos,qvel)
        # print('Reset Called')
        return self.get_observations()

    def get_joint_kinematics(self, exclude_com=False, concat=False):
        '''Returns qpos and qvel of the agent.'''
        qpos = np.copy(self.sim.data.qpos)
        qvel = np.copy(self.sim.data.qvel)
        if exclude_com:
            qpos = self._remove_by_indices(qpos, self._get_COM_indices())
            qvel = self._remove_by_indices(qvel, self._get_COM_indices())
        if concat:
            return np.concatenate([qpos, qvel]).flatten()
        return qpos, qvel

    # ----------------------------
    # Methods to override:
    # ----------------------------

    def _get_COM_indices(self):
        return [0,1,2]

    def _get_trunk_rot_joint_indices(self):
        return [3, 4, 5]

    def _get_not_actuated_joint_indices(self):
        return self._get_COM_indices() + [3,4,5]
    
    def _get_actuated_joint_indices(self):
        return [6, 7, 8, 9, 10, 11, 12, 13]

    def _get_max_actuator_velocities(self):
        """Maximum joint velocities approximated from the reference data."""
        return np.array([5, 1, 10, 10, 5, 1, 10, 10])

    def get_joint_indices_for_phase_estimation(self):
        # return both knee and hip joints
        return [6, 8, 10, 12]

    def has_ground_contact(self):
        has_contact = [False, False]
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if contact.geom1 == 0 and contact.geom2 == 4:
                # right foot has ground contact
                has_contact[1] = True
            elif contact.geom1 == 0 and contact.geom2 == 7:
                # left foot has ground contact
                has_contact[0] = True

        # if cfg.is_mod(cfg.MOD_3_PHASES):
        #     double_stance = all(has_contact)
        #     if cfg.is_mod(cfg.MOD_GRND_CONTACT_ONE_HOT):
        #         if double_stance:
        #             return [False, False, True]
        #         else:
        #             has_contact += [False]
        #     else: has_contact + [double_stance]

        # # when both feet have no ground contact
        # if cfg.is_mod(cfg.MOD_GROUND_CONTACT_NNS) and not any(has_contact):
        #     # print('Both feet without ground contact!')
        #     # let the left and right foot network handle this situation
        #     has_contact = np.array(has_contact)
        #     has_contact[:2] = True

        return has_contact