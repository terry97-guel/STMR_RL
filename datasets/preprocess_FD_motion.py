"""Retarget motions from keypoint (.txt) files."""
import os
import inspect

import time

import numpy as np
from pyquaternion import Quaternion

import isaacgym
from legged_gym.envs import *
import legged_gym.utils.kinematics.urdf as pk

from rsl_rl.datasets import pose3d
from pybullet_utils import transformations
import pybullet
import pybullet_data as pd

from datasets.retarget_utils import *


import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR

ROBOT = "go1"
ROBOT = ROBOT.lower()
MOTIONS = ['go1trot', 'hopturn', 'pace0', 'pace1', 'sidesteps', 'trot0', 'trot1']
MOTIONS = ['hopturn', 'pace0', 'pace1', 'sidesteps', 'trot0', 'trot1']
MOTIONS = ['pace0', 'pace1', 'trot0', 'trot1']
MOTIONS = ['videowalk0']

for MOTION in MOTIONS:
  # MR_LS = ['NMR', "SMR","STMR", 'TO']
  # MR_LS = ['TO']
  # MR_LS = ["NMR"]
  MR_LS = ["STMR"]

  for MR in MR_LS:
    class A1config:
      VISUALIZE_RETARGETING = True

      URDF_FILENAME = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/{ROBOT}/urdf/{ROBOT}.urdf"
      OUTPUT_DIR = f'{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_processed'

      if ROBOT == 'a1':
        INIT_POS = np.array([0, 0, 0.26])
        DEFAULT_JOINT_POSE = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
        INIT_ROT = np.array([0, 0, 0, 1.0])
      if ROBOT == "go1":
        INIT_POS = np.array([0, 0, 0.27])
        DEFAULT_JOINT_POSE = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
        INIT_ROT = np.array([0, 0, 0, 1.0])
      if ROBOT == "al":
        INIT_POS = np.array([0, 0, 0.40])
        DEFAULT_JOINT_POSE = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
        INIT_ROT = np.array([0, 0, 0, 1.0])

      FR_FOOT_NAME = "FR_foot"
      FL_FOOT_NAME = "FL_foot"
      HR_FOOT_NAME = "RR_foot"
      HL_FOOT_NAME = "RL_foot"

      filename = f"{MOTION}_{ROBOT}_{MR}"
      input_file = f"{LEGGED_GYM_ROOT_DIR}/datasets/{MOTION}/{ROBOT}/{MR}/{MOTION}_{ROBOT}_{MR}_raw.txt"

    config = A1config()

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    TAR_TOE_VEL_LOCAL_SIZE = 12

    DEFAULT_ROT = np.array([0, 0, 0, 1])
    FORWARD_DIR = np.array([1, 0, 0])

    GROUND_URDF_FILENAME = "plane_implicit.urdf"

    REF_PELVIS_JOINT_ID = 0
    REF_NECK_JOINT_ID = 3

    REF_TOE_JOINT_IDS = [10, 15, 19, 23]
    REF_HIP_JOINT_IDS = [6, 11, 16, 20]

    chain_foot_fl = pk.build_serial_chain_from_urdf(
        open(config.URDF_FILENAME).read(), config.FL_FOOT_NAME)
    chain_foot_fr = pk.build_serial_chain_from_urdf(
        open(config.URDF_FILENAME).read(), config.FR_FOOT_NAME)
    chain_foot_rl = pk.build_serial_chain_from_urdf(
        open(config.URDF_FILENAME).read(), config.HL_FOOT_NAME)
    chain_foot_rr = pk.build_serial_chain_from_urdf(
        open(config.URDF_FILENAME).read(), config.HR_FOOT_NAME)


    def build_markers(num_markers):
      marker_radius = 0.02

      markers = []
      for i in range(num_markers):
        if (i == REF_NECK_JOINT_ID) or (i == REF_PELVIS_JOINT_ID)\
            or (i in REF_HIP_JOINT_IDS):
          col = [0, 0, 1, 1]  # hip:BLUE
        elif (i in REF_TOE_JOINT_IDS):
          col = [1, 0, 0, 1]  # toe:RED
        else:
          col = [0, 1, 0, 1]  # other:GREEN

        virtual_shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE, radius=marker_radius, rgbaColor=col)
        body_id = pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=virtual_shape_id,
            basePosition=[0, 0, 0],
            useMaximalCoordinates=True)
        markers.append(body_id)

      return markers


    def get_joint_pose(pose):
      return pose[(POS_SIZE + ROT_SIZE):(POS_SIZE + ROT_SIZE + JOINT_POS_SIZE)]


    def get_tar_toe_pos_local(pose):
      return pose[(POS_SIZE + ROT_SIZE +
                  JOINT_POS_SIZE):(POS_SIZE + ROT_SIZE + JOINT_POS_SIZE +
                                    TAR_TOE_POS_LOCAL_SIZE)]


    def get_linear_vel(pose):
      return pose[(POS_SIZE + ROT_SIZE + JOINT_POS_SIZE +
                  TAR_TOE_POS_LOCAL_SIZE):(POS_SIZE + ROT_SIZE + JOINT_POS_SIZE +
                                            TAR_TOE_POS_LOCAL_SIZE +
                                            LINEAR_VEL_SIZE)]


    def get_angular_vel(pose):
      return pose[(POS_SIZE + ROT_SIZE + JOINT_POS_SIZE + TAR_TOE_POS_LOCAL_SIZE +
                  LINEAR_VEL_SIZE):(POS_SIZE + ROT_SIZE + JOINT_POS_SIZE +
                                    TAR_TOE_POS_LOCAL_SIZE + LINEAR_VEL_SIZE +
                                    ANGULAR_VEL_SIZE)]


    def get_joint_vel(pose):
      return pose[(POS_SIZE + ROT_SIZE + JOINT_POS_SIZE + TAR_TOE_POS_LOCAL_SIZE +
                  LINEAR_VEL_SIZE +
                  ANGULAR_VEL_SIZE):(POS_SIZE + ROT_SIZE + JOINT_POS_SIZE +
                                      TAR_TOE_POS_LOCAL_SIZE + LINEAR_VEL_SIZE +
                                      ANGULAR_VEL_SIZE + JOINT_VEL_SIZE)]


    def get_tar_toe_vel_local(pose):
      return pose[(POS_SIZE + ROT_SIZE + JOINT_POS_SIZE + TAR_TOE_POS_LOCAL_SIZE +
                  LINEAR_VEL_SIZE + ANGULAR_VEL_SIZE +
                  JOINT_VEL_SIZE):(POS_SIZE + ROT_SIZE + JOINT_POS_SIZE +
                                    TAR_TOE_POS_LOCAL_SIZE + LINEAR_VEL_SIZE +
                                    ANGULAR_VEL_SIZE + JOINT_VEL_SIZE +
                                    TAR_TOE_VEL_LOCAL_SIZE)]


    def set_root_pos(root_pos, pose):
      pose[0:POS_SIZE] = root_pos
      return


    def set_root_rot(root_rot, pose):
      pose[POS_SIZE:(POS_SIZE + ROT_SIZE)] = root_rot
      return


    def set_joint_pose(joint_pose, pose):
      pose[(POS_SIZE + ROT_SIZE):] = joint_pose
      return


    def set_maker_pos(marker_pos, marker_ids):
      num_markers = len(marker_ids)
      assert (num_markers == marker_pos.shape[0])

      for i in range(num_markers):
        curr_id = marker_ids[i]
        curr_pos = marker_pos[i]

        pybullet.resetBasePositionAndOrientation(curr_id, curr_pos, DEFAULT_ROT)

      return


    def set_foot_marker_pos(marker_pos, robot_idx, unique_ids=None):
      marker_pos = marker_pos.reshape(4, 3)
      new_unique_ids = []

      for foot_pos, unique_id in zip(marker_pos, unique_ids):
        if unique_id is not None:
          new_unique_ids.append(
              pybullet.addUserDebugLine(
                  lineFromXYZ=foot_pos - np.array([0.0, 0.0, 0.04]),
                  lineToXYZ=foot_pos + np.array([0.0, 0.0, 0.04]),
                  lineWidth=4,
                  replaceItemUniqueId=unique_id,
                  lineColorRGB=[1, 0, 0],
                  parentObjectUniqueId=robot_idx))
        else:
          new_unique_ids.append(
              pybullet.addUserDebugLine(
                  lineFromXYZ=foot_pos - np.array([0.0, 0.0, 0.04]),
                  lineToXYZ=foot_pos + np.array([0.0, 0.0, 0.04]),
                  lineWidth=4,
                  lineColorRGB=[1, 0, 0],
                  parentObjectUniqueId=robot_idx))
      return new_unique_ids




    def get_pose(robot, qpos):
      root_pos = qpos[0:3]
      root_rot = qpos[3:7]
      joint_pose = qpos[[10,11,12, 7,8,9, 16,17,18, 13,14,15]] # change FR, FL, RR, RL order to FL, FR, RL, RR
      tar_toe_pos_local = np.squeeze(
          np.concatenate([
              chain_foot_fl.forward_kinematics(joint_pose[:3]).get_matrix()[:, :3,
                                                                            3],
              chain_foot_fr.forward_kinematics(joint_pose[3:6]).get_matrix()[:, :3,
                                                                            3],
              chain_foot_rl.forward_kinematics(joint_pose[6:9]).get_matrix()[:, :3,
                                                                            3],
              chain_foot_rr.forward_kinematics(joint_pose[9:12]).get_matrix()[:, :3,
                                                                              3],
          ], axis=-1))

      pose = np.concatenate([root_pos, root_rot, joint_pose, tar_toe_pos_local])

      return pose


    def update_camera(robot):
      base_pos = np.array(pybullet.getBasePositionAndOrientation(robot)[0])
      [yaw, pitch, dist] = pybullet.getDebugVisualizerCamera()[8:11]
      pybullet.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
      return



    def get_frames(robot, qpos_np, FRAME_DURATION):
      num_frames = qpos_np.shape[0]

      time_between_frames = FRAME_DURATION

      for f in range(num_frames - 1):
        print(f"{f}/{num_frames}", end="\r")
        # Current robot pose.
        ref_joint_pos = qpos_np[f]
        # ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
        # ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)
        curr_pose = get_pose(robot, ref_joint_pos)
        set_pose(robot, curr_pose)

        # Next robot pose.
        next_ref_joint_pos = qpos_np[f + 1]
        next_pose = get_pose(robot, next_ref_joint_pos)

        if f == 0:
          pose_size = curr_pose.shape[
              -1] + LINEAR_VEL_SIZE + ANGULAR_VEL_SIZE + JOINT_POS_SIZE + TAR_TOE_VEL_LOCAL_SIZE
          new_frames = np.zeros([num_frames - 1, pose_size])

        # Linear velocity in base frame.
        del_linear_vel = np.array((get_root_pos(next_pose) -
                                  get_root_pos(curr_pose))) / time_between_frames
        r = pybullet.getMatrixFromQuaternion(get_root_rot(curr_pose))
        del_linear_vel = np.matmul(del_linear_vel, np.array(r).reshape(3, 3))

        # Angular velocity in base frame.
        curr_quat = get_root_rot(curr_pose)
        next_quat = get_root_rot(next_pose)
        diff_quat = Quaternion.distance(
            Quaternion(curr_quat[3], curr_quat[0], curr_quat[1], curr_quat[2]),
            Quaternion(next_quat[3], next_quat[0], next_quat[1], next_quat[2]))
        del_angular_vel = pybullet.getDifferenceQuaternion(
            get_root_rot(curr_pose), get_root_rot(next_pose))
        axis, _ = pybullet.getAxisAngleFromQuaternion(del_angular_vel)
        del_angular_vel = np.array(axis) * (diff_quat * 2) / time_between_frames
        # del_angular_vel = pybullet.getDifferenceQuaternion(get_root_rot(curr_pose), get_root_rot(next_pose))
        # del_angular_vel = np.array(pybullet.getEulerFromQuaternion(del_angular_vel)) / time_between_frames
        inv_init_rot = transformations.quaternion_inverse(config.INIT_ROT)
        _, base_orientation_quat_from_init = pybullet.multiplyTransforms(
            positionA=(0, 0, 0),
            orientationA=inv_init_rot,
            positionB=(0, 0, 0),
            orientationB=get_root_rot(curr_pose))
        _, inverse_base_orientation = pybullet.invertTransform(
            [0, 0, 0], base_orientation_quat_from_init)
        del_angular_vel, _ = pybullet.multiplyTransforms(
            positionA=(0, 0, 0),
            orientationA=(inverse_base_orientation),
            positionB=del_angular_vel,
            orientationB=(0, 0, 0, 1))

        joint_velocity = np.array(
            get_joint_pose(next_pose) -
            get_joint_pose(curr_pose)) / time_between_frames
        toe_velocity = np.array(
            get_tar_toe_pos_local(next_pose) -
            get_tar_toe_pos_local(curr_pose)) / time_between_frames

        if f==0:
          del_linear_vel = np.zeros_like(del_linear_vel)
          del_angular_vel = np.zeros_like(del_angular_vel)
          joint_velocity = np.zeros_like(joint_velocity)
          toe_velocity = np.zeros_like(toe_velocity)

        curr_pose = np.concatenate([
            curr_pose, del_linear_vel, del_angular_vel, joint_velocity, toe_velocity
        ])

        new_frames[f] = curr_pose

      new_frames[:, 0:2] -= new_frames[0, 0:2]

      return new_frames


    def main(qpos_np, FRAME_DURATION):
      p = pybullet
      # p.connect(p.GUI, options=f"--width=1920 --height=1080")
      if config.VISUALIZE_RETARGETING:
        p.connect(
            p.GUI,
            options="--width=1920 --height=1080 --mp4=\"test.mp4\" --mp4fps=60")
      else:
        p.connect(
            p.DIRECT,
            options="--width=1920 --height=1080 --mp4=\"test.mp4\" --mp4fps=60")
      p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
      pybullet.setAdditionalSearchPath(pd.getDataPath())

      output_dir = config.OUTPUT_DIR
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)

      pybullet.resetSimulation()
      pybullet.setGravity(0, 0, 0)

      ground = pybullet.loadURDF(GROUND_URDF_FILENAME)  # pylint: disable=unused-variable
      robot = pybullet.loadURDF(
          config.URDF_FILENAME,
          config.INIT_POS,
          config.INIT_ROT,
          flags=p.URDF_MAINTAIN_LINK_ORDER)
      # Set robot to default pose to bias knees in the right direction.
      set_pose(
          robot,
          np.concatenate(
              [config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))


      p.removeAllUserDebugItems()

      retarget_frames = get_frames(robot, qpos_np, FRAME_DURATION)

      output_file = os.path.join(output_dir, f"{config.filename}.txt")
      output_motion(retarget_frames, output_file, motion_weight=1, frame_duration=FRAME_DURATION)

      if config.VISUALIZE_RETARGETING:
        foot_line_unique_ids = [None] * 4
        linear_vel_unique_id = None
        angular_vel_unique_id = None

        f = 0

        num_frames = retarget_frames.shape[0]
        
        render_every = 5
        for repeat in range(1 * int(num_frames/render_every)):
          time_start = time.time()

          f_idx = f % num_frames
          print("Frame {:d}".format(f_idx))

          # ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
          # ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)

          pose = retarget_frames[f_idx]

          set_pose(robot, pose)
          foot_line_unique_ids = set_foot_marker_pos(
              get_tar_toe_pos_local(pose), robot, foot_line_unique_ids)
          linear_vel_unique_id = set_linear_vel_pos(
              get_linear_vel(pose), robot, linear_vel_unique_id)
          angular_vel_unique_id = set_angular_vel_pos(
              get_angular_vel(pose), robot, angular_vel_unique_id)

          update_camera(robot)
          p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
          f += render_every

          time_end = time.time()
          sleep_dur = FRAME_DURATION - (time_end - time_start)
          sleep_dur = max(0, sleep_dur)

          time.sleep(sleep_dur)
          #time.sleep(0.5) # jp hack

        p.removeAllUserDebugItems()

      pybullet.disconnect()

      return

    if __name__ == "__main__":
      from toolbox.read import read_json
      # motion_file = "datasets/hopturn/hopturn.txt"
      motion_dict = read_json(config.input_file)
      qpos_np = np.array(motion_dict['Frames'])
      qpos_np = qpos_np[:,:19]
      time_step = motion_dict['FrameDuration']
      main(qpos_np, time_step)
      # tf.app.run(main)
