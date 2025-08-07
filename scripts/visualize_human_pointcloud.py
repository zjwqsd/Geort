import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from geort.utils.config_utils import get_config
from geort.utils.path import get_human_data
from geort.dataset import MultiPointDataset
import matplotlib.cm as cm

def visualize_human_ik_training_with_animation(
    human_data_name='human_alex',
    hand='xhand_right',
    num_samples=2048,
    start_frame=0,
    num_frames=300
):
    # === 加载人手数据 ===
    human_data_path = get_human_data(human_data_name)
    human_data_raw = np.load(human_data_path)  # shape: [T, 21, 3]
    T = human_data_raw.shape[0]
    end_frame = min(start_frame + num_frames, T)
    human_data_clip = human_data_raw[start_frame:end_frame]  # [T_clip, 21, 3]

    # === 提取用于训练的关键指尖点云 ===
    config = get_config(hand)
    human_finger_idxes = [f["human_hand_id"] for f in config["fingertip_link"]]
    human_points = np.array([human_data_clip[:, idx, :3] for idx in human_finger_idxes])  # [N_finger, T_clip, 3]

    point_dataset = MultiPointDataset.from_points(human_points, n=num_samples)
    actual_samples = min(num_samples, len(point_dataset))
    sampled_points = np.array([point_dataset[i] for i in range(actual_samples)])  # [B, N_finger, 3]

    # === 设置指骨连接关系 ===
    finger_hierarchy = [
        [0, 1, 2, 3, 4],     # 大拇指
        [0, 5, 6, 7, 8],     # 食指
        [0, 9, 10, 11, 12],  # 中指
        [0, 13, 14, 15, 16], # 无名指
        [0, 17, 18, 19, 20]  # 小拇指
    ]

    # === 创建画布和元素 ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Human Hand Skeleton + IK Training Pointcloud Samples [{start_frame}:{end_frame}]")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-0.1, 0.2)
    ax.set_ylim(-0.1, 0.2)
    ax.set_zlim(-0.1, 0.2)

    # 关键点动画（每帧更新）
    scat = ax.scatter([], [], [], color='r', s=20)

    # 骨骼线动画（每帧更新）
    lines = []
    for _ in range(5):  # 5 根手指
        finger_lines = []
        for _ in range(4):  # 每根手指连接 4 个关节
            line, = ax.plot([0, 0], [0, 0], [0, 0], color="blue", linewidth=1)
            finger_lines.append(line)
        lines.append(finger_lines)

    # 点云静态可视化
    n_finger = sampled_points.shape[1]
    color_map = cm.get_cmap('tab10', n_finger)
    for i in range(n_finger):
        finger_points = sampled_points[:, i, :]  # shape: [N, 3]
        ax.scatter(finger_points[:, 0], finger_points[:, 1], finger_points[:, 2],
                   s=1, alpha=0.4, color=color_map(i), label=f"Finger {i}")

    ax.legend()

    # === 每一帧更新函数 ===
    def update(frame_idx):
        frame_data = human_data_clip[frame_idx]  # shape: [21, 3]
        xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
        scat._offsets3d = (xs, ys, zs)

        for f_idx, finger in enumerate(finger_hierarchy):
            for i in range(4):
                p_id = finger[i]
                c_id = finger[i + 1]
                px, py, pz = frame_data[p_id]
                cx, cy, cz = frame_data[c_id]
                lines[f_idx][i].set_data([px, cx], [py, cy])
                lines[f_idx][i].set_3d_properties([pz, cz])

        return scat, *[line for finger_lines in lines for line in finger_lines]

    ani = FuncAnimation(
        fig, update,
        frames=human_data_clip.shape[0],
        interval=100,
        repeat=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_human_ik_training_with_animation(
        human_data_name='rot_alex',
        hand='xhand_right',
        num_samples=2048,
        start_frame=0,
        num_frames=1000
    )
