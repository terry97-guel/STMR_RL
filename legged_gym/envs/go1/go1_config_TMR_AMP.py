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
from legged_gym.envs.go1.go1_config_Common import Cfg as GO1_Cfg
from legged_gym.envs.go1.go1_config_Common import CfgPPO as GO1_CfgPPO
from legged_gym.envs.go1.go1_config_Common import MOTION, ROBOT

MR = "TMR"
RL = "AMP"
MOTION_FILES = glob.glob(f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_processed/*')

class Cfg( GO1_Cfg ):
    class env( GO1_Cfg.env ):
        amp_motion_files = MOTION_FILES

    class rewards( GO1_Cfg.rewards ):
        class scales( GO1_Cfg.rewards.scales ):
            pos_motion     = 150
            ang_motion     = 150
            dof_pos_motion = 150

            dof_vel_motion = 50
            lin_vel_motion = 50
            ang_vel_motion = 50

class CfgPPO( GO1_CfgPPO ):
    class runner( GO1_CfgPPO.runner ):
        # experiment_name = f"AMP/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}"
        amp_motion_files = MOTION_FILES

