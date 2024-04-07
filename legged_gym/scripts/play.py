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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from isaacgym import gymtorch, gymapi, gymutil
NO_RAND = False

def play(args):
    register_tasks(args.task, args.seed, NO_RAND=NO_RAND)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    env_cfg.env.get_commands_from_joystick = False
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.env.reference_state_initialization = False

    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_com_displacement = False
    
    env_cfg.domain_rand.test_time = False

    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset(random_time=False)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, name = args.task+'.pt')
        print('Exported policy as jit script to: ', path)

    env.reset(random_time=False)
    obs = env.get_observations()
    
    # env.default_dof_pos[:] = torch.tensor([
    #     0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8
    # ])
    actions_ls = []
    obs_ls = []
    time_ls = []
    for repeat_n in range(100):
        for i in range(int(env.max_episode_length)):
            # env.reset()
            # env.update()
            
            # if env.times >= env.amp_loader.trajectory_lens[0] - env.dt:
            #     env.reset(random_time=False)
            #     obs = env.get_observations()
            #     # env.reset(random_time=True)

            obs_ls.append(obs[0].detach().cpu().numpy().flatten().tolist())
            actions = policy(obs.detach())
            actions_ls.append(actions[0].detach().cpu().numpy().flatten().tolist())
            time_ls.append(env.times[0])
            # obs, _, rews, dones, infos, _, _ = env.step(actions.detach(), RESET_ABLED=False)
            # actions = policy(obs.detach())
            # env.default_dof_pos[:] = torch.tensor([
            #     0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8
            # ])

            # actions = torch.ones(env_cfg.env.num_envs, 12) * torch.sin(torch.tensor(0.05 * i))
            # actions = torch.zeros_like(actions)
            
            # contact_info = env.gym.get_rigid_contacts(env.sim)     
            # contact_info = env.gym.get_env_rigid_contacts(env.envs[0])
            # actions = torch.zeros_like(actions)       
            obs, _, rews, dones, infos, _, _ = env.step(actions.detach(), RESET_ABLED=True)
            if dones.any():
                
                # from matplotlib import pyplot as plt
                # plt.plot(actions_ls)
                # plt.show()

                # logs_dict = {}
                # logs_dict['obs'] = obs_ls
                # logs_dict['actions'] = actions_ls
                # logs_dict["Timeframe"] = time_ls
                # # save as json
                # import json
                # with open('logs.json', 'w') as f:
                #     json.dump(logs_dict, f)
                actions_ls = []
                # obs_ls = []
                # time_ls = []
            # env.root_states[0][0] += 0.01
            # env_ids_int32 = torch.tensor([0]).to(dtype=torch.int32)
            # env.gym.set_actor_root_state_tensor_indexed(env.sim,
            #                                             gymtorch.unwrap_tensor(env.root_states[0]),
            #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            
            # env.update()
            # if env.reset_buf.any():
            #     for _ in range(1000):
            #         env.render()


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = True
    MOVE_CAMERA = False
    args = get_args()
    # args.task = "go1base_STMR_trot0"
    # args.seed = 1
    play(args)
