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

# %%

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from isaacgym import gymtorch, gymapi, gymutil

EXPORT_POLICY = True
RECORD_FRAMES = True
MOVE_CAMERA = False
# args = get_args()

# %%
import pickle 
from pathlib import Path
with open(LEGGED_GYM_ROOT_DIR/Path('datasets/default.pkl'), 'rb') as f:
    args = pickle.load(f)

from argparse import Namespace
args = Namespace(**args)
args.task = "go1base_STMR_go1trot"

ROBOT = args.task.split('_')[0]
MR = args.task.split('_')[1]
MOTION = args.task.split('_')[2]

# %%
register_tasks(args.task)
env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
# override some parameters for testing
env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
env_cfg.env.get_commands_from_joystick = False
env_cfg.terrain.num_rows = 5
env_cfg.terrain.num_cols = 5
env_cfg.terrain.curriculum = False
env_cfg.noise.add_noise = False

env_cfg.domain_rand.randomize_gains = False
env_cfg.domain_rand.randomize_base_mass = True
env_cfg.domain_rand.randomize_friction = True
env_cfg.domain_rand.randomize_restitution = True
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

# %%
device = 'cpu'
torch_model = ppo_runner.alg.actor_critic.to(device=device)
torch_model.forward = torch_model.act_inference
torch_input = torch.ones(1,env_cfg.env.num_observations).to(device=device)

torch_model(torch_input)

# %%
if 'base' in ROBOT:
    raw_robot_name = ROBOT.split("base")[0]
else:
    raw_robot_name = ROBOT

savename = f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{raw_robot_name}/{MR}/{MOTION}_{ROBOT}_{MR}.onnx'
onnx_program = torch.onnx.export(
    torch_model, 
    torch_input,
    savename,
    opset_version=9,
    input_names=["input"],
    output_names=["action"])


# %%
import onnxruntime as ort
import numpy as np


ort_sess = ort.InferenceSession(savename)
outputs = ort_sess.run(None, {'input': torch_input.numpy()})

outputs
# %%