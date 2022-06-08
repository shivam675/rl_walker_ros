#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
import tf
import numpy
import time
import math
import numpy as np
from std_msgs.msg import Float32MultiArray

"""
 wrenches:
      -
        force:
          x: -0.134995398774
          y: -0.252811705608
          z: -0.0861598399337
        torque:
          x: -0.00194729925705
          y: 0.028723398244
          z: -0.081229664152
    total_wrench:
      force:
        x: -0.134995398774
        y: -0.252811705608
        z: -0.0861598399337
      torque:
        x: -0.00194729925705
        y: 0.028723398244
        z: -0.081229664152
    contact_positions:
      -
        x: -0.0214808318267
        y: 0.00291348151391
        z: -0.000138379966267
    contact_normals:
      -
        x: 0.0
        y: 0.0
        z: 1.0
    depths: [0.000138379966266991]
  -
    info: "Debug:  i:(2/4)     my geom:monoped::lowerleg::lowerleg_contactsensor_link_collision_1\
  \   other geom:ground_plane::link::collision         time:50.405000000\n"
    collision1_name: "monoped::lowerleg::lowerleg_contactsensor_link_collision_1"
    collision2_name: "ground_plane::link::collision"

"""
"""
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
gazebo_msgs/ContactState[] states
  string info
  string collision1_name
  string collision2_name
  geometry_msgs/Wrench[] wrenches
    geometry_msgs/Vector3 force
      float64 x
      float64 y
      float64 z
    geometry_msgs/Vector3 torque
      float64 x
      float64 y
      float64 z
  geometry_msgs/Wrench total_wrench
    geometry_msgs/Vector3 force
      float64 x
      float64 y
      float64 z
    geometry_msgs/Vector3 torque
      float64 x
      float64 y
      float64 z
  geometry_msgs/Vector3[] contact_positions
    float64 x
    float64 y
    float64 z
  geometry_msgs/Vector3[] contact_normals
    float64 x
    float64 y
    float64 z
  float64[] depths
"""

class WalkerState(object):

    def __init__(self, max_height, min_height, abs_max_roll, abs_max_pitch, joint_increment_value = 0.05, done_reward = -1000.0, alive_reward=10.0, desired_force=7.08, desired_yaw=0.0, weight_r1=1.0, weight_r2=1.0, weight_r3=1.0, weight_r4=1.0, weight_r5=1.0, discrete_division=10):
        rospy.logdebug("Starting Catbot State Class object...")
        self.desired_world_point = Vector3(0.0, 0.0, 0.0)
        self._min_height = min_height
        self._max_height = max_height
        self._abs_max_roll = abs_max_roll
        self._abs_max_pitch = abs_max_pitch
        self._joint_increment_value = joint_increment_value
        self._done_reward = done_reward
        self._alive_reward = alive_reward
        self._desired_force = desired_force
        self._desired_yaw = desired_yaw
        self.base_linear_vel = None
        self._weight_r1 = weight_r1
        self._weight_r2 = weight_r2
        self._weight_r3 = weight_r3
        self._weight_r4 = weight_r4
        self._weight_r5 = weight_r5

        self.step_time = time.time()

        self.previous_obs = np.array([
            0, 0, 0.78, 
            0, 0, 0, 
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0,            
            0, 0, 0, 0, 0,            
        ], dtype=np.float32)


        self.previous_action = np.array([
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ], np.float32)


        # self._list_of_observations = ["distance_from_desired_point",
        #          "base_roll",
        #          "base_pitch",
        #          "base_yaw",
        #          "contact_force_left_leg",
        #          "contact_force_right_leg",
        #          "joint_states_bum_zlj",
        #          "joint_states_bum_xlj",
        #          "joint_states_bum_ylj",
        #          "joint_states_knee_left",
        #          "joint_states_foot_lj",
                 
        #          "joint_states_bum_zrj",
        #          "joint_states_bum_xrj",
        #          "joint_states_bum_yrj",
        #          "joint_states_knee_right",
        #          "joint_states_foot_rj",]

        self._list_of_observations = [
                # body 3d pose
                "body_pose_x",
                "body_pose_y",
                "body_pose_z",

                # body linear vel
                "body_vel_x",
                "body_vel_y",
                "body_vel_z",

                # body orientation 
                "base_roll",
                "base_pitch",
                "base_yaw",

                # body odom angular vels
                "body_vel_angular_x",
                "body_vel_angular_y",
                "body_vel_angular_z",
                # Angles
                "joint_states_bum_zlj",
                "joint_states_bum_xlj",
                "joint_states_bum_ylj",
                "joint_states_knee_left",
                "joint_states_foot_lj",
                "joint_states_bum_zrj",
                "joint_states_bum_xrj",
                "joint_states_bum_yrj",
                "joint_states_knee_right",
                "joint_states_foot_rj",

                # anglur vels 
                "angular_vel_bum_zlj",
                "angular_vel_bum_xlj",
                "angular_vel_bum_ylj",
                "angular_vel_knee_left",
                "angular_vel_foot_lj",
                "angular_vel_bum_zrj",
                "angular_vel_bum_xrj",
                "angular_vel_bum_yrj",
                "angular_vel_knee_right",
                "angular_vel_foot_rj",

                # distance
                "distance_from_desired_point",

                # contact forces of sensors
                "contact_force_left_leg",
                "contact_force_right_leg",

                # 10 action commands in previous time step
                "prev_action_bum_zlj",
                "prev_action_bum_xlj",
                "prev_action_bum_ylj",
                "prev_action_knee_left",
                "prev_action_foot_lj",
                "prev_action_bum_zrj",
                "prev_action_bum_xrj",
                "prev_action_bum_yrj",
                "prev_action_knee_right",
                "prev_action_foot_rj",
                
                ]

        self._discrete_division = discrete_division
        # We init the observation ranges and We create the bins now for all the observations

        self.base_position = Point()
        self.base_orientation = Quaternion()
        self.base_linear_acceleration = Vector3()
        self.left_contact_force = Vector3()
        self.right_contact_force = Vector3()
        self.joints_state = JointState()

        # Odom we only use it for the height detection and planar position ,
        #  because in real robots this data is not trivial.

        self.obs_pub_object = rospy.Publisher('/obs', Float32MultiArray, queue_size= 2)
        self.action_pub_object = rospy.Publisher('/action_per_step', Float32MultiArray, queue_size= 2)

        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        # We use the IMU for orientation and linearacceleration detection
        rospy.Subscriber("/imu/data", Imu, self.imu_callback)
        # We use it to get the contact force, to know if its in the air or stumping too hard.
        rospy.Subscriber("/lower_left_leg_contactsensor_state", ContactsState, self.left_contact_callback)
        rospy.Subscriber("/lower_right_leg_contactsensor_state", ContactsState, self.right_contact_callback)
        # We use it to get the joints positions and calculate the reward associated to it
        rospy.Subscriber("/joint_states", JointState, self.joints_state_callback)

    def check_all_systems_ready(self):
        """
        We check that all systems are ready
        :return:
        """
        data_pose = None
        while data_pose is None and not rospy.is_shutdown():
            try:
                data_pose = rospy.wait_for_message("/odom", Odometry, timeout=0.1)
                self.base_position = data_pose.pose.pose.position
                rospy.logdebug("Current odom READY")
            except:
                rospy.logdebug("Current odom pose not ready yet, retrying for getting robot base_position")

        imu_data = None
        while imu_data is None and not rospy.is_shutdown():
            try:
                imu_data = rospy.wait_for_message("/imu/data", Imu, timeout=0.1)
                self.base_orientation = imu_data.orientation
                self.base_linear_acceleration = imu_data.linear_acceleration
                rospy.logdebug("Current imu_data READY")
            except:
                rospy.logdebug("Current imu_data not ready yet, retrying for getting robot base_orientation, and base_linear_acceleration")

        left_contacts_data = None

        #####################################################################################
        ########################### Check both legs #########################################

        while left_contacts_data is None and not rospy.is_shutdown():
            try:
                left_contacts_data = rospy.wait_for_message("/lower_left_leg_contactsensor_state", ContactsState, timeout=0.1)
                for state in left_contacts_data.states:
                    self.left_contact_force = state.total_wrench.force
                rospy.logdebug("Current LEFT contacts_data READY")
            except:
                rospy.logdebug("Current LEFT contacts_data not ready yet, retrying")

        right_contacts_data = None
        while right_contacts_data is None and not rospy.is_shutdown():
            try:
                right_contacts_data = rospy.wait_for_message("/lower_right_leg_contactsensor_state", ContactsState, timeout=0.1)
                for state in right_contacts_data.states:
                    self.right_contact_force = state.total_wrench.force
                rospy.logdebug("Current RIGHT contacts_data READY")
            except:
                rospy.logdebug("Current RIGHT contacts_data not ready yet, retrying")

        ##########################################################################################
        ##########################################################################################



        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message("/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                rospy.logdebug("Current joint_states not ready yet, retrying==>"+str(e))

        rospy.logdebug("ALL SYSTEMS READY")





    def set_desired_world_point(self, x, y, z):
        """
        Point where you want the Monoped to be
        :return:
        """
        self.desired_world_point.x = x
        self.desired_world_point.y = y
        self.desired_world_point.z = z


    def get_base_height(self):
        return abs(self.base_position.z)

    def get_base_rpy(self):
        euler_rpy = Vector3()
        euler = tf.transformations.euler_from_quaternion(
            [self.base_orientation.x, self.base_orientation.y, self.base_orientation.z, self.base_orientation.w])

        euler_rpy.x = euler[0]
        euler_rpy.y = euler[1]
        euler_rpy.z = euler[2]
        return euler_rpy

    def get_distance_from_point(self, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((self.base_position.x, self.base_position.y, self.base_position.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def get_left_contact_force_magnitude(self):
        """
        You will see that because the X axis is the one pointing downwards, it will be the one with
        higher value when touching the floor
        For a Robot of total mas of 0.55Kg, a gravity of 9.81 m/sec**2, Weight = 0.55*9.81=5.39 N
        Falling from around 5centimetres ( negligible height ), we register peaks around
        Fx = 7.08 N
        :return:
        """
        contact_force = self.left_contact_force
        contact_force_np = numpy.array((contact_force.x, contact_force.y, contact_force.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude

    
    def get_right_contact_force_magnitude(self):
        """
        You will see that because the X axis is the one pointing downwards, it will be the one with
        higher value when touching the floor
        For a Robot of total mas of 0.55Kg, a gravity of 9.81 m/sec**2, Weight = 0.55*9.81=5.39 N
        Falling from around 5centimetres ( negligible height ), we register peaks around
        Fx = 7.08 N
        :return:
        """
        contact_force = self.right_contact_force
        contact_force_np = numpy.array((contact_force.x, contact_force.y, contact_force.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude

    # def get_base_linear_vel(self):
    #     return self.base_linear_vel
    
    # def get_base_angular_vel(self):
    #     return self.base_angular_vel

    def get_joint_states(self):
        return self.joints_state
    

    def odom_callback(self,msg):
        self.base_position = msg.pose.pose.position
        self.base_linear_vel = msg.twist.twist.linear
        self.base_angular_vel = msg.twist.twist.angular

    def imu_callback(self,msg):
        self.base_orientation = msg.orientation
        self.base_linear_acceleration = msg.linear_acceleration

    def left_contact_callback(self,msg):
        """
        /lowerleg_contactsensor_state/states[0]/contact_positions ==> PointContact in World
        /lowerleg_contactsensor_state/states[0]/contact_normals ==> NormalContact in World

        ==> One is an array of all the forces, the other total,
         and are relative to the contact link referred to in the sensor.
        /lowerleg_contactsensor_state/states[0]/wrenches[]
        /lowerleg_contactsensor_state/states[0]/total_wrench
        :param msg:
        :return:
        """
        for state in msg.states:
            self.left_contact_force = state.total_wrench.force


    def right_contact_callback(self,msg):
        """
        /lowerleg_contactsensor_state/states[0]/contact_positions ==> PointContact in World
        /lowerleg_contactsensor_state/states[0]/contact_normals ==> NormalContact in World

        ==> One is an array of all the forces, the other total,
         and are relative to the contact link referred to in the sensor.
        /lowerleg_contactsensor_state/states[0]/wrenches[]
        /lowerleg_contactsensor_state/states[0]/total_wrench
        :param msg:
        :return:
        """
        for state in msg.states:
            self.right_contact_force = state.total_wrench.force


    

    def joints_state_callback(self,msg):
        self.joints_state = msg

    def catbot_height_ok(self):

        height_ok = self._min_height <= self.get_base_height() < self._max_height
        # print(self.get_base_height())
        return height_ok

    def catbot_orientation_ok(self):

        orientation_rpy = self.get_base_rpy()
        roll_ok = self._abs_max_roll > abs(orientation_rpy.x)
        pitch_ok = self._abs_max_pitch > abs(orientation_rpy.y)
        orientation_ok = roll_ok and pitch_ok
        return orientation_ok



#######################################################################################################
#######################################################################################################
#######################################################################################################
################################ REWARD CODE ##########################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################





    # def calculate_reward_joint_position(self, weight=1.0):
    #     """
    #     We calculate reward base on the joints configuration. The more near 0 the better.
    #     :return:
    #     """
    #     acumulated_joint_pos = 0.0
    #     for joint_pos in self.joints_state.position:
    #         # Abs to remove sign influence, it doesnt matter the direction of turn.
    #         acumulated_joint_pos += abs(joint_pos)
    #         rospy.logdebug("calculate_reward_joint_position>>acumulated_joint_pos=" + str(acumulated_joint_pos))
    #     reward = weight * acumulated_joint_pos
    #     rospy.logdebug("calculate_reward_joint_position>>reward=" + str(reward))
    #     return reward

    def calculate_velocity(self, weight=1.0):
        a = self.base_linear_vel.x
        r1, r2, r3 = 0, 0, 0
        if self.desired_world_point.x > 0:
            r1 = (weight*a)/self.desired_world_point.x
        if self.desired_world_point.y > 0:
            r2 = (weight*a)/self.desired_world_point.y
        if self.desired_world_point.z > 0:
            r3 = (weight*a)/self.desired_world_point.z

        # m = Vector3()
        # pass
        
        return r1 + r2 +r3

    def calculate_reward_joint_effort(self, weight=1.0):
        """
        We calculate reward base on the joints effort readings. The more near 0 the better.
        :return:
        """
        acumulated_joint_effort = 0.0
        for joint_effort in self.joints_state.effort:
            # Abs to remove sign influence, it doesnt matter the direction of the effort.
            acumulated_joint_effort += abs(joint_effort)
            rospy.logdebug("calculate_reward_joint_effort>>joint_effort=" + str(joint_effort))
            rospy.logdebug("calculate_reward_joint_effort>>acumulated_joint_effort=" + str(acumulated_joint_effort))
        reward = weight * acumulated_joint_effort
        rospy.logdebug("calculate_reward_joint_effort>>reward=" + str(reward))
        return reward

    def calculate_left_reward_contact_force(self, weight=1.0):
        """
        We calculate reward base on the contact force.
        The nearest to the desired contact force the better.
        We use exponential to magnify big departures from the desired force.
        Default ( 7.08 N ) desired force was taken from reading of the robot touching
        the ground from a negligible height of 5cm.
        :return:
        """
        force_magnitude = self.get_left_contact_force_magnitude()
        force_displacement = force_magnitude - self._desired_force

        rospy.logdebug("calculate_left_reward_contact_force>>force_magnitude=" + str(force_magnitude))
        rospy.logdebug("calculate_left_reward_contact_force>>force_displacement=" + str(force_displacement))
        # Abs to remove sign
        reward = weight * abs(force_displacement)
        rospy.logdebug("calculate_left_reward_contact_force>>reward=" + str(reward))
        return reward

    def calculate_right_reward_contact_force(self, weight=1.0):
        """
        We calculate reward base on the contact force.
        The nearest to the desired contact force the better.
        We use exponential to magnify big departures from the desired force.
        Default ( 7.08 N ) desired force was taken from reading of the robot touching
        the ground from a negligible height of 5cm.
        :return:
        """
        force_magnitude = self.get_right_contact_force_magnitude()
        force_displacement = force_magnitude - self._desired_force

        rospy.logdebug("calculate_right_reward_contact_force>>force_magnitude=" + str(force_magnitude))
        rospy.logdebug("calculate_right_reward_contact_force>>force_displacement=" + str(force_displacement))
        # Abs to remove sign
        reward = weight * abs(force_displacement)
        rospy.logdebug("calculate_right_reward_contact_force>>reward=" + str(reward))
        return reward


    def calculate_reward_orientation(self, weight=1.0):
        """
        We calculate the reward based on the orientation.
        The more its closser to 0 the better because it means its upright
        desired_yaw is the yaw that we want it to be.
        to praise it to have a certain orientation, here is where to set it.
        :return:
        """
        curren_orientation = self.get_base_rpy()
        yaw_displacement = curren_orientation.z - self._desired_yaw
        rospy.logdebug("calculate_reward_orientation>>[R,P,Y]=" + str(curren_orientation))
        acumulated_orientation_displacement = abs(curren_orientation.x) + abs(curren_orientation.y) + abs(yaw_displacement)
        reward = weight * acumulated_orientation_displacement
        rospy.logdebug("calculate_reward_orientation>>reward=" + str(reward))
        return reward

    def calculate_reward_distance_from_des_point(self, weight=1.0):
        """
        We calculate the distance from the desired point.
        The closser the better
        :param weight:
        :return:reward
        """
        distance = self.get_distance_from_point(self.desired_world_point)
        reward = weight * distance
        rospy.logdebug("calculate_reward_orientation>>reward=" + str(reward))
        return reward

    def calculate_total_reward(self):
        """
        We consider VERY BAD REWARD -7 or less
        Perfect reward is 0.0, and total reward 1.0.
        The defaults values are chosen so that when the robot has fallen or very extreme joint config:
        r1 = -8.04
        r2 = -8.84
        r3 = -7.08
        r4 = -10.0 ==> We give priority to this, giving it higher value.
        :return:
        """

        r1 = self.calculate_velocity(self._weight_r1)
        r2 = self.calculate_reward_joint_effort(self._weight_r2)
        # Desired Force in Newtons, taken form idle contact with 9.81 gravity.

        r3_a = self.calculate_left_reward_contact_force(self._weight_r3)
        r3_b = self.calculate_right_reward_contact_force(self._weight_r3)
        r4 = self.calculate_reward_orientation(self._weight_r4)
        r5 = self.calculate_reward_distance_from_des_point(self._weight_r5)

        # The sign depend on its function.
        total_reward = self._alive_reward - r2 - r3_a - r3_b- r4 - r5 + r1

        rospy.logdebug("###############")
        rospy.logdebug("alive_bonus=" + str(self._alive_reward))
        rospy.logdebug("r1 joint_position=" + str(r1))
        rospy.logdebug("r2 joint_effort=" + str(r2))
        rospy.logdebug("r3a&b contact_force=" + str(r3_a) + " and " + str(r3_b))
        rospy.logdebug("r4 orientation=" + str(r4))
        rospy.logdebug("r5 distance=" + str(r5))
        rospy.logdebug("total_reward=" + str(total_reward))
        rospy.logdebug("###############")

        return total_reward

#######################################################################################################
#######################################################################################################



    def calculate_angular_vels(self):
        my_list = list(self.previous_obs)
        joint_states = self.get_joint_states()
        joint_states = joint_states.position
        angular_vels = []
        for i in range(12, 22):
            tn = time.time() - self.step_time
            val = (joint_states[i-12] - my_list[i])/tn
            angular_vels.append(val)

        return angular_vels
            



#######################################################################################################
#######################################################################################################
#######################################################################################################
########################################## Obsservation code ##########################################
#######################################################################################################
#######################################################################################################



    def get_observations(self):
        """
        Returns the state of the robot needed for OpenAI QLearn Algorithm
        The state will be defined by an array of the:
        1) distance from desired point in meters
        2) The pitch orientation in radians
        3) the Roll orientation in radians
        4) the Yaw orientation in radians
        5) Force in contact sensor in Newtons
        6-7-8) State of the 3 joints in radians

        observation = [distance_from_desired_point,
                 base_roll,
                 base_pitch,
                 base_yaw,
                 contact_force,
                 joint_states_haa,
                 joint_states_hfe,
                 joint_states_kfe]

        :return: observation
        """

        distance_from_desired_point = self.get_distance_from_point(self.desired_world_point)

        ################################################
        base_orientation = self.get_base_rpy()
        base_roll = base_orientation.x
        base_pitch = base_orientation.y
        base_yaw = base_orientation.z
        #################################################
        body_pose_x = self.base_position.x
        body_pose_y = self.base_position.y
        body_pose_z = self.base_position.z
        #################################################
        body_vel_x = self.base_linear_vel.x
        body_vel_y = self.base_linear_vel.y
        body_vel_z = self.base_linear_vel.z
        #################################################
        body_vel_angular_x = self.base_angular_vel.x
        body_vel_angular_y = self.base_angular_vel.y
        body_vel_angular_z = self.base_angular_vel.z
        #################################################
        
        list_of_angular_vels = self.calculate_angular_vels()
        angular_vel_bum_zlj = list_of_angular_vels[0] 
        angular_vel_bum_xlj = list_of_angular_vels[1]
        angular_vel_bum_ylj = list_of_angular_vels[2]
        angular_vel_knee_left = list_of_angular_vels[3]
        angular_vel_foot_lj = list_of_angular_vels[4]
        angular_vel_bum_zrj = list_of_angular_vels[5]
        angular_vel_bum_xrj = list_of_angular_vels[6]
        angular_vel_bum_yrj = list_of_angular_vels[7]
        angular_vel_knee_right = list_of_angular_vels[8]
        angular_vel_foot_rj = list_of_angular_vels[9]
        
        #################################################

        list_of_previous_actions = list(self.previous_action)

        prev_action_bum_zlj = list_of_previous_actions[0]
        prev_action_bum_xlj = list_of_previous_actions[1]
        prev_action_bum_ylj = list_of_previous_actions[2]
        prev_action_knee_left = list_of_previous_actions[3]
        prev_action_foot_lj = list_of_previous_actions[4]
        prev_action_bum_zrj = list_of_previous_actions[5]
        prev_action_bum_xrj = list_of_previous_actions[6]
        prev_action_bum_yrj = list_of_previous_actions[7]
        prev_action_knee_right = list_of_previous_actions[8]
        prev_action_foot_rj = list_of_previous_actions[9]

        #################################################
        left_contact_force = self.get_left_contact_force_magnitude()
        right_contact_force = self.get_right_contact_force_magnitude()

        joint_states = self.get_joint_states()
        # joint_states_haa = joint_states.position[0]
        # joint_states_hfe = joint_states.position[1]
        # joint_states_kfe = joint_states.position[2]

        joint_states_bum_zlj = joint_states.position[0] 
        joint_states_bum_xlj = joint_states.position[1]
        joint_states_bum_ylj = joint_states.position[2]
        joint_states_knee_left = joint_states.position[3]
        joint_states_foot_lj = joint_states.position[4]
        joint_states_bum_zrj = joint_states.position[5]
        joint_states_bum_xrj = joint_states.position[6]
        joint_states_bum_yrj = joint_states.position[7]
        joint_states_knee_right = joint_states.position[8]
        joint_states_foot_rj = joint_states.position[9]

        observation = []
        for obs_name in self._list_of_observations:
            if obs_name == "distance_from_desired_point":
                observation.append(distance_from_desired_point)
            
            elif obs_name == "base_roll":
                observation.append(base_roll)
            elif obs_name == "base_pitch":
                observation.append(base_pitch)
            elif obs_name == "base_yaw":
                observation.append(base_yaw)
            
            elif obs_name == "contact_force_left_leg":
                observation.append(left_contact_force)
            elif obs_name == "contact_force_right_leg":
                observation.append(right_contact_force)
            
            elif obs_name == "joint_states_bum_zlj":
                observation.append(joint_states_bum_zlj)
            elif obs_name == "joint_states_bum_xlj":
                observation.append(joint_states_bum_xlj)
            elif obs_name == "joint_states_bum_ylj":
                observation.append(joint_states_bum_ylj)
            elif obs_name == "joint_states_knee_left":
                observation.append(joint_states_knee_left)
            elif obs_name == "joint_states_foot_lj":
                observation.append(joint_states_foot_lj)
            
            elif obs_name == "joint_states_bum_zrj":
                observation.append(joint_states_bum_zrj)
            elif obs_name == "joint_states_bum_xrj":
                observation.append(joint_states_bum_xrj)
            elif obs_name == "joint_states_bum_yrj":
                observation.append(joint_states_bum_yrj)
            elif obs_name == "joint_states_knee_right":
                observation.append(joint_states_knee_right)
            elif obs_name == "joint_states_foot_rj":
                observation.append(joint_states_foot_rj)

            elif obs_name == "body_pose_x":
                observation.append(body_pose_x)
            elif obs_name == "body_pose_y":
                observation.append(body_pose_y)
            elif obs_name == "body_pose_z":
                observation.append(body_pose_z)
            
            elif obs_name == "body_vel_x":
                observation.append(body_vel_x)
            elif obs_name == "body_vel_y":
                observation.append(body_vel_y)
            elif obs_name == "body_vel_z":
                observation.append(body_vel_z)

            elif obs_name == "body_vel_angular_x":
                observation.append(body_vel_angular_x)
            elif obs_name == "body_vel_angular_y":
                observation.append(body_vel_angular_y)
            elif obs_name == "body_vel_angular_z":
                observation.append(body_vel_angular_z)
            
            elif obs_name == "angular_vel_bum_zlj":
                observation.append(angular_vel_bum_zlj)
            elif obs_name == "angular_vel_bum_xlj":
                observation.append(angular_vel_bum_xlj)
            elif obs_name == "angular_vel_bum_ylj":
                observation.append(angular_vel_bum_ylj)
            elif obs_name == "angular_vel_knee_left":
                observation.append(angular_vel_knee_left)
            elif obs_name == "angular_vel_foot_lj":
                observation.append(angular_vel_foot_lj)
            

            elif obs_name == "angular_vel_bum_zrj":
                observation.append(angular_vel_bum_zrj)
            elif obs_name == "angular_vel_bum_xrj":
                observation.append(angular_vel_bum_xrj)
            elif obs_name == "angular_vel_bum_yrj":
                observation.append(angular_vel_bum_yrj)
            elif obs_name == "angular_vel_knee_right":
                observation.append(angular_vel_knee_right)
            elif obs_name == "angular_vel_foot_rj":
                observation.append(angular_vel_foot_rj)

            

            elif obs_name == "prev_action_bum_zlj":
                observation.append(prev_action_bum_zlj)
            elif obs_name == "prev_action_bum_xlj":
                observation.append(prev_action_bum_xlj)
            elif obs_name == "prev_action_bum_ylj":
                observation.append(prev_action_bum_ylj)
            elif obs_name == "prev_action_knee_left":
                observation.append(prev_action_knee_left)
            elif obs_name == "prev_action_foot_lj":
                observation.append(prev_action_foot_lj)
            
            elif obs_name == "prev_action_bum_zrj":
                observation.append(prev_action_bum_zrj)
            elif obs_name == "prev_action_bum_xrj":
                observation.append(prev_action_bum_xrj)
            elif obs_name == "prev_action_bum_yrj":
                observation.append(prev_action_bum_yrj)
            elif obs_name == "prev_action_knee_right":
                observation.append(prev_action_knee_right)
            elif obs_name == "prev_action_foot_rj":
                observation.append(prev_action_foot_rj)

            else:
                print(obs_name)
                raise NameError('Observation Asked does not exist=='+str(obs_name))
        # print(len(observation))
        msg = Float32MultiArray()
        msg.data = observation
        self.obs_pub_object.publish(msg)
        return observation






    def get_action_to_position(self, action, prev_obs):
        """
        Here we have the ACtions number to real joint movement correspondance.

        ################ REF MSG ###########################
        header: 
            seq: 19580
            stamp: 
                secs: 392
                nsecs: 602000000
        frame_id: ''
            name:
            - bum_xlj
            - bum_xrj
            - bum_ylj
            - bum_yrj
            - bum_zlj
            - bum_zrj
            - foot_lj
            - foot_rj
            - knee_left
            - knee_right
        ####################################################

        :param action: Integer that goes from 0 to 5, because we have 6 actions.
        :return:
        """
        self.step_time = time.time()
        self.previous_obs = prev_obs
        self.previous_action = action
        # We get current Joints values
        joint_states = self.get_joint_states()
        joint_states_position = joint_states.position


        action_position = [
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            ]

        rospy.logdebug("get_action_to_position>>>"+str(joint_states_position))
        try: #Increment ankle_lj
            for idx, val in enumerate(action_position):
                action_position[idx] = joint_states_position[idx]
        except:
            pass    
        msg = Float32MultiArray()
        msg.data = action_position
        self.action_pub_object.publish(msg)
        return action_position

    def process_data(self):
        """
        We return the total reward based on the state in which we are in and if its done or not
        ( it fell basically )
        :return: reward, done
        """
        catbot_height_ok = self.catbot_height_ok()
        catbot_orientation_ok = self.catbot_orientation_ok()
        # print(catbot_height_ok, catbot_orientation_ok)

        done = not(catbot_height_ok and catbot_orientation_ok)
        if done:
            rospy.logdebug("It fell, so the reward has to be very low")
            total_reward = self._done_reward
        else:
            rospy.logdebug("Calculate normal reward because it didn't fall.")
            total_reward = self.calculate_total_reward()

        return total_reward, done

    def testing_loop(self):

        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.calculate_total_reward()
            rate.sleep()


# if __name__ == "__main__":
#     rospy.init_node('monoped_state_node', anonymous=True)
#     monoped_state = CatbotState(max_height=3.0,
#                                  min_height=0.6,
#                                  abs_max_roll=0.7,
#                                  abs_max_pitch=0.7)
#     monoped_state.testing_loop()
