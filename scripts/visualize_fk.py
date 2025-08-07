import numpy as np
import time
import torch
from sapien.render import RenderMaterial
from geort.utils.config_utils import get_config
from geort.env.hand import HandKinematicModel
from geort.model import FKModel
from geort.formatter import HandFormatter
import sapien.core as sapien
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, required=True, help='Hand name (e.g. xhand_right, allegro_right)')
    return parser.parse_args()

def main():
    args = parse_args()
    hand_name = args.hand
    config = get_config(hand_name)
    hand = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = hand.get_viewer_env()

    # 关键点初始化
    info = config["fingertip_link"]
    link_names = [f["link"] for f in info]
    offsets = [f["center_offset"] for f in info]
    hand.initialize_keypoint(link_names, offsets)

    # 创建材质（红：GT，绿：FK预测）
    red = RenderMaterial();  red.base_color = [1.0, 0.0, 0.0, 1.0]
    green = RenderMaterial(); green.base_color = [0.0, 1.0, 0.0, 1.0]

    # 创建 marker（红色：GT，绿色：预测）
    gt_markers, pred_markers = [], []
    for _ in link_names:
        builder1 = hand.scene.create_actor_builder()
        builder1.add_sphere_visual(radius=0.007, material=red)
        gt_markers.append(builder1.build_static(name="gt"))

        builder2 = hand.scene.create_actor_builder()
        builder2.add_sphere_visual(radius=0.007, material=green)
        pred_markers.append(builder2.build_static(name="pred"))

    # 初始化 FK 模型
    keypoint_joints = [ [config["joint_order"].index(j) for j in f["joint"]] for f in info ]
    q_lower, q_upper = hand.get_joint_limit()
    formatter = HandFormatter(q_lower, q_upper)
    fk_model = FKModel(keypoint_joints=keypoint_joints).cuda().eval()

    # 加载权重
    ckpt_path = f"checkpoint/fk_model_{hand_name}.pth"
    fk_model.load_state_dict(torch.load(ckpt_path))
    print(f"✅ Loaded FK checkpoint from {ckpt_path}")

    # 初始化
    q_current = np.random.uniform(q_lower, q_upper)
    hand.set_qpos_target(q_current)

    while True:
        q_target = np.random.uniform(q_lower, q_upper)

        for alpha in np.linspace(0, 1, 100):
            q_interp = (1 - alpha) * q_current + alpha * q_target
            hand.set_qpos_target(q_interp)

            for _ in range(3):
                viewer_env.update()

            # ✅ ground truth
            for i, link in enumerate(hand.keypoint_links):
                link_pose = link.get_pose()
                offset = hand.keypoint_offsets[i]
                world_pos = link_pose.p + link_pose.to_transformation_matrix()[:3, :3] @ offset
                gt_markers[i].set_pose(sapien.Pose(world_pos))

            # ✅ FK 网络预测
            q_normed = formatter.normalize_torch(torch.tensor(q_interp).unsqueeze(0).float().cuda())
            with torch.no_grad():
                pred = fk_model(q_normed).squeeze(0).cpu().numpy()  # shape: [N_keypoints, 3]

            # FK 预测是在 base_link 坐标系下，我们需要加上 base_link 的世界坐标
            base_pose = hand.base_link.get_pose()
            R = base_pose.to_transformation_matrix()[:3, :3]
            t = base_pose.p

            for i in range(len(pred)):
                pred_world = t + R @ pred[i]
                pred_markers[i].set_pose(sapien.Pose(pred_world))

        q_current = q_target

if __name__ == "__main__":
    main()
