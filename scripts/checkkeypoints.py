import numpy as np
import time
from sapien.render import RenderMaterial
from geort.utils.config_utils import get_config
from geort.env.hand import HandKinematicModel
import sapien.core as sapien

def main():
    hand_name = "xhand_right"  # 或 "xhand_right"

    config = get_config(hand_name)
    hand = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = hand.get_viewer_env()

    # 初始化 keypoints
    info = config["fingertip_link"]
    link_names = [f["link"] for f in info]
    offsets = [f["center_offset"] for f in info]
    hand.initialize_keypoint(link_names, offsets)

    # 创建红色材质
    red_material = RenderMaterial()
    red_material.base_color = [1.0, 0.0, 0.0, 1.0]

    # 创建 marker 球体
    markers = []
    for _ in link_names:
        builder = hand.scene.create_actor_builder()
        builder.add_sphere_visual(radius=0.004, material=red_material)
        marker = builder.build_static(name="keypoint_marker")
        markers.append(marker)

    print(f"\n🎯 正在展示 `{hand_name}` 动态缓慢运动，红点实时贴合 keypoints\n")

    # 初始化起始姿态
    q_current = np.random.uniform(hand.joint_lower_limit, hand.joint_upper_limit)
    hand.set_qpos_target(q_current)

    while True:
        # 生成目标姿态
        q_target = np.random.uniform(hand.joint_lower_limit, hand.joint_upper_limit)

        # 缓慢插值过渡（100 步，约 2 秒）
        for alpha in np.linspace(0, 1, 100):
            q_interp = (1 - alpha) * q_current + alpha * q_target
            hand.set_qpos_target(q_interp)

            # 等待驱动到位
            for _ in range(3):
                viewer_env.update()

            # 实时更新 keypoint 红点位置
            for i, link in enumerate(hand.keypoint_links):
                link_pose = link.get_pose()
                offset = hand.keypoint_offsets[i]
                world_pos = link_pose.p + link_pose.to_transformation_matrix()[:3, :3] @ offset
                markers[i].set_pose(sapien.Pose(world_pos))

        q_current = q_target

if __name__ == "__main__":
    main()
