# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import glob

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

MOTION = "hopturn"
ROBOT = "go1"
ROBOT = ROBOT.lower()

friction1 = 0.2
friction2 = 0.4

class Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 5480
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 40
        num_privileged_obs = 46
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        # amp_motion_files = MOTION_FILES
        ee_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        get_commands_from_joystick = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.27] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'FL_thigh_joint': 0.9,     # [rad]
            'FL_calf_joint': -1.8,   # [rad]

            'FR_hip_joint': 0.0,  # [rad]
            'FR_thigh_joint': 0.9,     # [rad]
            'FR_calf_joint': -1.8,  # [rad]

            'RL_hip_joint': 0.0,   # [rad]
            'RL_thigh_joint': 0.9,   # [rad]
            'RL_calf_joint': -1.8,    # [rad]
            
            'RR_hip_joint': -0.0,   # [rad]
            'RR_thigh_joint': 0.9,   # [rad]
            'RR_calf_joint': -1.8,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 6

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
        curriculum = False

    class asset( LeggedRobotCfg.asset ):
        file = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/{ROBOT}/urdf/{ROBOT}.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = [
            "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
            "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        
    class domain_rand:
        test_time = False
        randomize_friction = True
        friction_range = [friction1, friction2]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05
            height_measurements = 0.1

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            pos_motion     = 150
            ang_motion     = 150
            dof_pos_motion = 150

            dof_vel_motion = 50
            lin_vel_motion = 50
            ang_vel_motion = 50

            termination = 0.0
            tracking_lin_vel = 0  
            tracking_ang_vel = 0  
            lin_vel_z = 0.0      # penalize vertical velocity           
            ang_vel_xy = 0.0     # penalize horizontal angular velocity
            orientation = 0.0    # penalize orientation error            
            torques = -0.0002     # penalize torques                        
            dof_vel = -0.0001        # penalize joint velocities               
            dof_acc = 0.0        # penalize joint accelerations               
            base_height = 0.0    # penalize base height                               
            feet_air_time =  0.0 # penalize feet air time                          
            collision = 0.0      # penalize collisions                   
            feet_stumble = 0.0   # penalize feet stumble                    
            action_rate = 0.0    # penalize change in action                 
            stand_still = 0.0    # penalize standing still                     
            dof_pos_limits = 0.0 # penalize joint position limits             
            
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0, 0] # min max [m/s]
            lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

class CfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = f"AMP/{MOTION}/{ROBOT}/TMR/{MOTION}_{ROBOT}_TMR_{friction1}_{friction2}"
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 50_000 # number of policy updates

        amp_reward_coef = 2
        # amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.01, 0.01, 0.01] * 4
        # resume = True


