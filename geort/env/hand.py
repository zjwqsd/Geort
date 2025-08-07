import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import sapien.core as sapien
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from geort.utils.config_utils import get_config, save_json
from geort.utils.hand_utils import get_entity_by_name, get_active_joints, get_active_joint_indices
from datetime import datetime
from tqdm import tqdm 
from pathlib import Path 
import math

class HandKinematicModel:
    def __init__(
        self,
        scene=None,
        render=False,
        hand=None,
        hand_urdf="",
        n_hand_dof=16,
        base_link="base_link",
        joint_names=[],
        kp=200.0,
        kd=10,
        **kwargs
    ):
        if scene is None:
            scene = sapien.Scene()
            scene.set_timestep(1 / 100.0)
        self.scene = scene
        self.renderer = None

        if hand is not None:
            self.hand = hand
        else:
            material = scene.create_physical_material(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            )
            loader = scene.create_urdf_loader()
            loader.fix_root_link = True
            # loader.collision_group = 0
            loader.default_material = material
            self.hand = loader.load(hand_urdf)
            self.hand.set_root_pose(sapien.Pose([0, 0, 0.35], [0.695, 0, -0.718, 0]))

        self.pmodel = self.hand.create_pinocchio_model()

        self.base_link = get_entity_by_name(self.hand.get_links(), base_link)
        self.base_link_idx = self.hand.get_links().index(self.base_link)

        self.all_joints = get_active_joints(self.hand, joint_names)
        all_limits = [joint.get_limits() for joint in self.all_joints]

        self.joint_names = joint_names
        self.user_idx_to_sim_idx = get_active_joint_indices(self.hand, joint_names)
        print("User-to-Sim Joint", self.user_idx_to_sim_idx)
        self.sim_idx_to_user_idx = [
            self.user_idx_to_sim_idx.index(i) for i in range(len(self.user_idx_to_sim_idx))
        ]
        print("Sim-to-User Joint", self.sim_idx_to_user_idx)



        self.joint_lower_limit = np.zeros(len(joint_names))
        self.joint_upper_limit = np.zeros(len(joint_names))

        clip_ratio_dict = kwargs.get("joint_range_clip_ratio", {})

        for i, (joint_name, limit) in enumerate(zip(joint_names, all_limits)):
            urdf_lower = limit[0][0]
            urdf_upper = limit[0][1]

            if joint_name in clip_ratio_dict:
                ratio_low, ratio_high = clip_ratio_dict[joint_name]
                assert 0.0 <= ratio_low < ratio_high <= 1.0, f"Invalid clip ratio for joint {joint_name}"
                clipped_lower = urdf_lower + ratio_low * (urdf_upper - urdf_lower)
                clipped_upper = urdf_lower + ratio_high * (urdf_upper - urdf_lower)
                self.joint_lower_limit[i] = clipped_lower
                self.joint_upper_limit[i] = clipped_upper
                print(f"🔧 Clip {joint_name}: {urdf_lower:.3f}~{urdf_upper:.3f} → {clipped_lower:.3f}~{clipped_upper:.3f}")
            else:
                self.joint_lower_limit[i] = urdf_lower
                self.joint_upper_limit[i] = urdf_upper

        init_qpos = self.convert_user_order_to_sim_order(
            (self.joint_lower_limit + self.joint_upper_limit) / 2
        )
        self.hand.set_qpos(init_qpos)
        self.hand.set_qvel(0.0 * init_qpos)
        self.qpos_target = init_qpos

        for i, joint in enumerate(self.all_joints):
            print(i, self.joint_names[i], joint, self.joint_lower_limit[i], self.joint_upper_limit[i])
            joint.set_drive_property(kp, kd, force_limit=1000)

        print("\n🔍 验证 joint 顺序映射")
        for i, joint in enumerate(self.all_joints):
            print(f"  Joint[{i}]: {joint.get_name()} ==? joint_order[{i}]: {self.joint_names[i]}")


    def __del__(self):
        if hasattr(self, "scene"):
            del self.scene

    def get_n_dof(self):
        return len(self.joint_lower_limit)

    def get_joint_limit(self):
        return self.joint_lower_limit, self.joint_upper_limit

    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
        keypoint_links = [get_entity_by_name(self.hand.get_links(), link) for link in keypoint_link_names]
        print(keypoint_links)

        keypoint_links_id_dict = {
            link_name: (self.hand.get_links().index(keypoint_links[i]), i)
            for i, link_name in enumerate(keypoint_link_names)
        }
        self.keypoint_links = keypoint_links
        self.keypoint_links_id_dict = keypoint_links_id_dict
        self.keypoint_offsets = np.array(keypoint_offsets)

    def convert_user_order_to_sim_order(self, qpos):
        return qpos[self.sim_idx_to_user_idx]

    def keypoint_from_qpos(self, qpos, ret_vec=False):
        qpos = self.convert_user_order_to_sim_order(qpos)
        self.pmodel.compute_forward_kinematics(qpos)
        base_pose = self.pmodel.get_link_pose(self.base_link_idx)

        result = {}
        vec_result = []

        for m, (link_idx, i) in self.keypoint_links_id_dict.items():
            pose = self.pmodel.get_link_pose(link_idx)
            new_pose = sapien.Pose(
                p=pose.p + (pose.to_transformation_matrix()[:3, :3] @ self.keypoint_offsets[i].reshape(3, 1)).reshape(-1),
                q=pose.q,
            )
            x = (base_pose.inv() * new_pose).p
            vec_result.append(x)
            result[m] = x

        if ret_vec:
            return np.array(vec_result)
        return result


    @staticmethod
    def build_from_config(config, **kwargs):
        render = kwargs.get("render", False)
        urdf_path = config["urdf_path"]
        n_hand_dof = len(config["joint_order"])
        base_link = config["base_link"]
        joint_order = config["joint_order"]
        joint_range_clip_ratio = config.get("joint_range_clip_ratio", {})

        model = HandKinematicModel(
            hand_urdf=urdf_path,
            render=render,
            n_hand_dof=n_hand_dof,
            base_link=base_link,
            joint_names=joint_order,
            joint_range_clip_ratio=joint_range_clip_ratio
        )
        return model

    def get_viewer_env(self):
        return HandViewerEnv(self)

    def get_scene(self):
        return self.scene

    def get_renderer(self):
        return self.renderer

    def set_qpos_target(self, qpos):
        qpos = np.clip(qpos, self.joint_lower_limit + 1e-3, self.joint_upper_limit - 1e-3)
        self.qpos_target = qpos.copy()
        
        for i, joint in enumerate(self.all_joints):
            joint.set_drive_target(self.qpos_target[i])

    def precise_fk_tensor(self, qpos_tensor: torch.Tensor) -> torch.Tensor:
        qpos_np = qpos_tensor.detach().cpu().numpy()
        keypoints_batch = []
        for qpos in qpos_np:
            keypoints = self.keypoint_from_qpos(qpos, ret_vec=True)  # [N_keypoint, 3]
            keypoints_batch.append(keypoints)
        keypoints_batch = np.stack(keypoints_batch, axis=0)  # [B, N, 3]
        return torch.from_numpy(keypoints_batch).to(qpos_tensor.device).float()




class HandViewerEnv:
    def __init__(self, model):
        scene = model.get_scene()
        scene.set_timestep(1 / 100.0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        scene.add_ground(altitude=0)

        viewer = scene.create_viewer()
        viewer.set_scene(scene)
        viewer.set_camera_xyz(x=-2, y=0, z=2.5)
        viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=0)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        self.model = model
        self.scene = scene
        self.viewer = viewer

    def update(self):
        self.scene.step()
        self.scene.update_render()
        self.viewer.render()


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, default="allegro")
    args = parser.parse_args()

    config = get_config(args.hand)
    model = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = model.get_viewer_env()

    
    print("\n🔧 检查 right_hand_thumb_bend_joint 驱动器配置：")

    for joint in model.all_joints:
        if joint.get_name() == "right_hand_thumb_bend_joint":
            print(f"🧩 Joint: {joint.get_name()}")
            print(f"     Type: {joint.get_type()}")
            print(f"     Drive target: {joint.get_drive_target().flatten()[0]:.4f}")
            print(f"     kp: {joint.get_stiffness():.2f}")
            print(f"     kd: {joint.get_damping():.2f}")
            print(f"     force_limit: {joint.get_force_limit():.2f}")


    n_dof = model.get_n_dof()
    dof_lower, dof_upper = model.get_joint_limit()

    steps = 0
    joint_names = config["joint_order"]

    fingertip_info = config.get("fingertip_link", [])

    print("\n🧪 开始逐根手指最大运动测试...\n")

    for finger in fingertip_info:
        finger_name = finger["name"]
        joint_list = finger["joint"]

        print(f"\n🖐️ 测试手指：{finger_name}")
        for joint_name in joint_list:
            print(f"")
            if joint_name not in joint_names:
                print(f"  ⚠️ 未在 joint_order 中找到关节 {joint_name}")
                continue
            print(f"test joint {joint_name}")
            user_idx = joint_names.index(joint_name)
            q = dof_lower.copy()
            num_steps = 100
            for step in range(num_steps):
                interpolated_q = dof_lower + (dof_upper - dof_lower) * (step / (num_steps - 1))
                q[user_idx] = interpolated_q[user_idx]
                model.set_qpos_target(q)
                for _ in range(3):
                    viewer_env.update()

            q[user_idx] = dof_upper[user_idx]
            model.set_qpos_target(q)
            for _ in range(100):
                viewer_env.update()

    q = dof_lower.copy()

    num_steps = 100

    for step in range(num_steps):
        interpolated_q = dof_lower + (dof_upper - dof_lower) * (step / (num_steps - 1))
        q[:] = interpolated_q
        model.set_qpos_target(q)
        for _ in range(3):
            viewer_env.update()

    q[:] = dof_upper
    model.set_qpos_target(q)
    for _ in range(100):
        viewer_env.update()

    for step in range(num_steps):
        interpolated_q = dof_upper - (dof_upper - dof_lower) * (step / (num_steps - 1))
        q[:] = interpolated_q
        model.set_qpos_target(q)
        for _ in range(3):
            viewer_env.update()

    q[:] = dof_lower
    model.set_qpos_target(q)
    for _ in range(100):
        viewer_env.update()


    print("\n🎲 进入随机运动测试模式，请关闭窗口或按 Ctrl+C 退出程序。")

    while True:
        # 随机采样一个 joint 配置，范围在 [dof_lower, dof_upper]
        q = np.random.uniform(dof_lower, dof_upper)
        model.set_qpos_target(q)

        # 渐变过渡到该姿态
        for _ in range(30):  # 过渡步数越多越平滑
            viewer_env.update()

