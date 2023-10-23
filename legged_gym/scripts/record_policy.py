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

import cv2
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.test_time = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    env.reset(random_time=False)
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    camera_rot = 0
    camera_rot_per_sec = np.pi / 6
    img_idx = 0

    video_duration = 10
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')
    video = None

    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 360
    camera_properties.height = 240
    h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
    
    # for i in range(num_frames):
    env.reset(random_time=False)
    while True:
        if env.times >= env.amp_loader.trajectory_lens[0] - env.dt:
            break
        actions = policy(obs.detach())
        obs, _, _, _, infos, _, _ = env.step(actions.detach())

        # Reset camera position.
        look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
        camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
        camera_relative_position = 1.2 * np.array([np.cos(camera_rot), np.sin(camera_rot), 0.45])
        cam_pos = look_at + camera_relative_position
        
        cam_pos = gymapi.Vec3(cam_pos[0], cam_pos[1], cam_pos[2])
        cam_target = gymapi.Vec3(look_at[0], look_at[1], look_at[2])
        env.gym.set_camera_location(h1, env.envs[0], cam_pos, cam_target)


        if RECORD_FRAMES:
            frames_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
            if not os.path.isdir(frames_path):
                os.makedirs(frames_path, exist_ok=True)
            # filename = os.path.join('logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            if video is None:
                video = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
            video.write(img)
            img_idx += 1 

    video.release()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = get_args()
    args.task = "go1_TMR_AMP"
    args.headless = True
    args.use_gpu_pipeline = False
    args.sim_device='cpu'
    args.rl_device='cpu'
    from pyvirtualdisplay.smartdisplay import SmartDisplay
    SCREEN_CAPTURE_RESOLUTION = (1027, 768)
    virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
    virtual_display.start()
    play(args)