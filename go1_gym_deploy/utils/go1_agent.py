
from go1_gym_deploy.utils.go1_state_estimator import GO1StateEstimator
from go1_gym_deploy.utils.cfg import Cfg
import numpy as np
import torch
import lcm
import time

from go1_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt


lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


class Go1HardwareAgent():
    def __init__(self):
        self.se = GO1StateEstimator(lc)
        self.se.spin()
        
        self.cfg = Cfg()
        self.timestep = 0
        
        # constants
        self.dt = self.cfg.control.decimation * self.cfg.sim.dt
        self.num_obs = self.cfg.env.num_observations
        self.num_envs = 1
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.num_actions = self.cfg.env.num_actions
        self.num_commands = self.cfg.commands.num_commands
        self.device = 'cpu' # ?
        
        
        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", ]
        
        self.default_dof_pos = np.array([
            self.cfg.init_state.default_joint_angles[name] for name in joint_names])

        self.obs_scales = self.cfg.normalization.obs_scales

        self.p_gains = self.cfg.control.stiffness['joint'] * np.ones(12)
        self.d_gains = self.cfg.control.damping['joint']   * np.ones(12)
            
        # Buffer
        self.actions = torch.zeros(12)
        self.time = time.time()
        # self.deploy_time = torch.zeros(1)
        # self.saved_deploy_time = torch.zeros(1)
        # self.start_time = time.time()
        
        self.obs = np.zeros(self.cfg.env.num_observations)
        self.obs_history = np.zeros((self.cfg.env.include_history_steps, self.cfg.env.num_observations))

        self.reset()

    
    def get_obs(self):
        return self.obs_history.flatten()

    def compute_obs(self):
        projected_gravity = self.se.get_gravity_vector()
        
        dof_pos = self.se.get_dof_pos()
        dof_vel = self.se.get_dof_vel()
        
        # clip_actions = self.cfg.normalization.clip_actions
        # self.actions = torch.clip(self.actions, -clip_actions, clip_actions).to(self.device) 
        assert (np.abs(self.actions)<self.cfg.normalization.clip_actions).all()
        deploy_time = self.timestep * self.dt
        
        obs = np.concatenate([
            projected_gravity,
            (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            dof_vel * self.obs_scales.dof_vel,
            self.actions,
            np.array([deploy_time]),
        ])
    
        self.dof_pos = dof_pos
        self.dof_vel = dof_vel
    
        self.obs = obs.copy()

    def insert_obs(self, obs):
        self.obs_history[:-1] = self.obs_history[1:]
        self.obs_history[-1] = obs

    def publish_action_(self, action, motion_q):
        joint_pos_tar = action * self.cfg.control.action_scale
        # joint_pos_tar += self.default_dof_pos
        joint_pos_tar += motion_q
        joint_vel_tar = np.zeros(12)
        torques = (joint_pos_tar - self.dof_pos) * self.p_gains + (joint_vel_tar - self.dof_vel) * self.d_gains
        
        self.joint_pos_tar = joint_pos_tar
        self.joint_vel_tar = joint_vel_tar
        self.torques       = torques
        
        self.publish_joint_target_(joint_pos_tar, joint_vel_tar)
        
    def publish_joint_target_(self, joint_pos_tar, joint_vel_tar=np.zeros(12), p_gains=None, d_gains=None):
        joint_pos_tar = joint_pos_tar[self.se.joint_idxs_inv]
        joint_vel_tar = joint_vel_tar[self.se.joint_idxs_inv]
        
        command_for_robot = pd_tau_targets_lcmt()
        command_for_robot.q_des = joint_pos_tar
        command_for_robot.qd_des = joint_vel_tar
        
        if p_gains is None:
            command_for_robot.kp = self.p_gains
        else:
            command_for_robot.kp = p_gains * np.ones(12)
        
        if d_gains is None:
            command_for_robot.kd = self.d_gains
        else:
            command_for_robot.kd = d_gains * np.ones(12)
            
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0
        lc.publish("pd_plustau_targets", command_for_robot.encode())
        
    def reset(self):
        self.actions[:] = 0
        self.timestep = 0
        self.time = time.time()
        self.start_time = time.time()
        self.compute_obs()

    def get_time(self):
        # return time.time() - self.start_time
        return self.timestep * self.dt
    
    def step(self, actions, motion_q):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions.flatten(), -clip_actions, clip_actions)
        self.actions = self.actions.detach().cpu().numpy()
        self.publish_action_(self.actions, motion_q)
        
        sleep_time = self.dt - (time.time()-self.time)
        if sleep_time < 0:
            print(f"Warning: sleep time is negative: {sleep_time}")
        else:
            time.sleep(sleep_time)
        self.time = time.time()
        
        self.compute_obs()
        obs = self.obs
        self.insert_obs(obs)
        obs_history = self.get_obs()
        
        self.timestep += 1
        return obs_history
        