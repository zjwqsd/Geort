import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, required=True, help="Hand name (e.g. allegro_right, xhand_right)")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points sampled per finger")
    return parser.parse_args()

def visualize_colored_robot_pointcloud(hand_name: str, num_points: int = 2048):
    # 构造路径并加载 .npz 文件
    npz_path = os.path.join('data', f'{hand_name}.npz')
    data = np.load(npz_path, allow_pickle=True)
    keypoint_dict = data['keypoint'].item()

    finger_names = list(keypoint_dict.keys())
    n_fingers = len(finger_names)
    color_map = cm.get_cmap('tab10', n_fingers)

    # 初始化图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, name in enumerate(finger_names):
        points = keypoint_dict[name]
        if points.shape[0] < num_points:
            selected_idx = np.random.choice(points.shape[0], points.shape[0], replace=False)
        else:
            selected_idx = np.random.choice(points.shape[0], num_points, replace=False)
        sampled = points[selected_idx]

        color = color_map(i)[:3]
        ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2],
                   s=2, alpha=0.7, color=color, label=name)

    ax.set_title(f"Finger-wise Colored Pointcloud: {hand_name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    visualize_colored_robot_pointcloud(args.hand, args.num_points)
