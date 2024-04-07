# %%
NO_RAND = True

GET_ALL = False
fps = 30
EXTRACT_VIDEO = fps is not None

import mujoco
from mujoco_viewer.mujoco_viewer import MujocoViewer
import numpy as np
from numpy.linalg import norm

from toolbox.resources import get_asset_dict, ASSET_SRC
from mjmr.util import reset, get_mr_info, get_xml_path, plot_contact_schedule, get_vel_contact_boolean, get_mujoco_contact_boolean,get_vel_contact_boolean_from_pos, get_key_id, output_amp_motion
from mjmr.task.Quadruped.info import QuadrupedRetargetInfo as RetargetInfo

from MotionBO.MotionBO import compute_cost

import copy
import time
from pathlib import Path
import shutil


    
def main(MR, seed, ROBOT, MOTION):
    from matplotlib import pyplot as plt

    xml_path = get_xml_path(ROBOT)
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    data  = mujoco.MjData(model)
    reset(model, data)
    homepose = data.qpos.copy()

    try:
        viewer.close()
    except Exception:
        pass

    viewer = MujocoViewer(
    model,data,mode='window',title="MPC",
    width=1200,height=800,hide_menus=True
    )
    mr_info   = RetargetInfo(model, data)

    if MOTION == 'sidesteps':
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -3
        viewer.cam.distance = 1.8
        viewer.cam.lookat = homepose[:3]

    if MOTION == 'hopturn':
        viewer.cam.azimuth = 180 + 45
        viewer.cam.elevation = -3
        viewer.cam.distance = 1.8
        viewer.cam.lookat = homepose[:3]

    import cv2
    def grab_image(viewer, resize_rate=None,interpolation=cv2.INTER_NEAREST):
        """
            Grab the rendered iamge
        """
        img = np.zeros((viewer.viewport.height,viewer.viewport.width,3),dtype=np.uint8)
        mujoco.mjr_render(viewer.viewport,viewer.scn,viewer.ctx)
        mujoco.mjr_readPixels(img, None,viewer.viewport,viewer.ctx)
        img = np.flipud(img) # flip image
        # Resize
        if resize_rate is not None:
            h = int(img.shape[0]*resize_rate)
            w = int(img.shape[1]*resize_rate)
            img = cv2.resize(img,(w,h),interpolation=interpolation)
        return img

    from mjpc.task.Quadruped.play import plot_skeleton
    from mjpc.task.Quadruped.info import QuadrupedMPCInfo as MPCInfo

    mpc_info = MPCInfo(model, data)

    from legged_gym import LEGGED_GYM_ROOT_DIR
    from fastdtw import fastdtw
    import numpy as np
    from scipy.spatial.distance import cityblock
    # load motion
    ROBOT_base = ROBOT + 'base'

    if NO_RAND:
        RAND = "STMR"
    else:
        RAND = "RAND"

    if GET_ALL:
        path = Path(LEGGED_GYM_ROOT_DIR)/f"performance/{RAND}/{MOTION}/{ROBOT_base}/{MR}/{MOTION}_{ROBOT_base}_{MR}/seed{seed}/pose_all.json"
    else:
        path = Path(LEGGED_GYM_ROOT_DIR)/f"performance/{RAND}/{MOTION}/{ROBOT_base}/{MR}/{MOTION}_{ROBOT_base}_{MR}/seed{seed}/pose_1k.json"
    import json
    # load json
    with open(path) as json_file:
        motion_data = json.load(json_file)

    target_qpos_array = np.array(motion_data['target'])
    deploy_qpos_array = np.array(motion_data['deploy'])

    target_qpos_array.shape, deploy_qpos_array.shape

    reset(model,data)

    site_ids = [
        mr_info.id.trunk_site,
        mr_info.id.FR_hip_site, mr_info.id.FR_calf_site, mr_info.id.FR_foot_site,
        mr_info.id.FL_hip_site, mr_info.id.FL_calf_site, mr_info.id.FL_foot_site,
        mr_info.id.RR_hip_site, mr_info.id.RR_calf_site, mr_info.id.RR_foot_site,
        mr_info.id.RL_hip_site, mr_info.id.RL_calf_site, mr_info.id.RL_foot_site,
        ]

    # site_ids = [
    #     mr_info.id.trunk_site,
    #     mr_info.id.FR_hip_site, mr_info.id.FR_calf_site,
    #     mr_info.id.FL_hip_site, mr_info.id.FL_calf_site,
    #     mr_info.id.RR_hip_site, mr_info.id.RR_calf_site,
    #     mr_info.id.RL_hip_site, mr_info.id.RL_calf_site,
    #     ]

    mujoco_idx = [3,4,5,6, 7,8,9, 10,11,12, 13,14,15, 16,17,18]
    amp_idx    = [6,3,4,5, 10,11,12, 7,8,9, 16,17,18, 13,14,15]

    target_qpos_total = np.array(motion_data['target'])
    deploy_qpos_total = np.array(motion_data['deploy'])

    target_qpos_total[:,:,mujoco_idx] = target_qpos_total[:,:,amp_idx].copy()
    deploy_qpos_total[:,:,mujoco_idx] = deploy_qpos_total[:,:,amp_idx].copy()

    target_qpos_total[:,:, 7:] +=  model.qpos0[7:] 
    deploy_qpos_total[:,:, 7:] +=  model.qpos0[7:]

    model_number = len(target_qpos_total)
    frame_number = len(target_qpos_total[0])

    print(model_number)

    if GET_ALL:
        render_every = 10
    elif EXTRACT_VIDEO:
        dt = 0.005 * 6
        render_every = int(1/fps/dt)
    else:
        render_every = 1
    key_point_error_ls = []
    for model_i in range(model_number):
        # model_i = -1
        target_qpos_array = target_qpos_total[model_i]
        deploy_qpos_array = deploy_qpos_total[model_i]

        target_site_ls = []
        deploy_site_ls = []
        for frame_i in range(frame_number):

            for _ in range(1):
                target_qpos = target_qpos_array[frame_i]
                deploy_qpos = deploy_qpos_array[frame_i]

                data.qpos[:] = target_qpos
                mujoco.mj_forward(model, data)
                for site_i in site_ids:
                    target_site_ls.append(data.site_xpos[site_i].copy())
                # viewer.render()

                if frame_i%render_every == 0:
                    plot_skeleton(model, data, mpc_info, viewer, rgba = [1,0,0,0.5])

                data.qpos[:] = deploy_qpos
                mujoco.mj_forward(model, data)
                for site_i in site_ids:
                    deploy_site_ls.append(data.site_xpos[site_i].copy())
                if frame_i%render_every == 0:
                    viewer.render()    

                if not GET_ALL and EXTRACT_VIDEO:
                    image_dir = Path(LEGGED_GYM_ROOT_DIR)/f"performance/video/{MOTION}/{ROBOT_base}/{RAND}/{MR}/{seed}/image"
                    image_name = image_dir/f"image_{frame_i}.png"
                    image_name.parent.mkdir(exist_ok=True, parents=True)
                    plt.figure()
                    plt.axis('off')
                    image = grab_image(viewer)
                    plt.imsave(image_name, image)
        if not GET_ALL and EXTRACT_VIDEO:
            image_files = [path for path in image_dir.iterdir() if path.name.endswith(".png")]
            idx_ls = [int(path.stem.split('_')[-1]) for path in image_files]
            idx_array = np.array(idx_ls)
            idx_sorted = np.argsort(idx_array)
            image_files_sorted = [image_files[idx] for idx in idx_sorted]

            output_file = image_dir/f"../{MOTION}_{ROBOT}_{MR}.mp4"
            import imageio
            # Create the video using imageio
            with imageio.get_writer(output_file, mode="I", fps=fps) as writer:
                for image_file in image_files_sorted:
                    frame = imageio.imread(image_file)
                    writer.append_data(frame)
            print(f"Video created: {output_file}")


        target_site_array = np.array(target_site_ls).reshape(frame_number, -1)
        deploy_site_array = np.array(deploy_site_ls).reshape(frame_number, -1)

        distance, path = fastdtw(target_site_array, deploy_site_array, dist=cityblock)
        key_point_error = distance/frame_number/len(site_ids)

        # key_point_error = np.mean(np.abs(np.array(target_site_ls) - np.array(deploy_site_ls)), axis=1)
        # key_point_error = np.max(key_point_error, axis=0)
        
        key_point_error_ls.append(key_point_error)
        print(key_point_error)
        # break
        
    # L1 loss between target and deploy site
    from matplotlib import pyplot as plt
    plt.plot(key_point_error_ls)
    viewer.close()

# MR = "STMR"
seed = 1
# MOTION = 'sidesteps'

for MOTION in ["trot1"]:
    for MR in ["NMR", "STMR", "AMP", "TO"]:
        for ROBOT in ['go1', "a1", 'al']:
            main(MR, seed, ROBOT, MOTION)
# %%
