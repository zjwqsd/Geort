import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", type=str, required=True, help="human hand data name (e.g. rot_alex, ref_alex)")
    return parser.parse_args()

def main():
    args = parse_args()

    # 构造数据路径
    hand_name = args.human
    data_path = os.path.join("data", f"{hand_name}.npy")
    data = np.load(data_path)
    num_frames = data.shape[0]

    # 手指的层级关系：根部 + 5 根手指，每根手指 4 个节点（包含根部）
    finger_hierarchy = [
        [0, 1, 2, 3, 4],    # 大拇指
        [0, 5, 6, 7, 8],    # 食指
        [0, 9, 10, 11, 12], # 中指
        [0, 13, 14, 15, 16],# 无名指
        [0, 17, 18, 19, 20] # 小拇指
    ]

    # 初始化画布
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Hand Skeleton: {hand_name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-0.1, 0.2)
    ax.set_ylim(-0.1, 0.2)
    ax.set_zlim(-0.1, 0.2)

    # 初始化关键点和连线
    scat = ax.scatter([], [], [], color='r', s=20)
    lines = []
    for _ in range(5):
        finger_lines = []
        for _ in range(4):
            line, = ax.plot([0, 0], [0, 0], [0, 0], color="blue", linewidth=1)
            finger_lines.append(line)
        lines.append(finger_lines)

    # 更新函数
    def update(frame):
        frame_data = data[frame]
        xs, ys, zs = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
        scat._offsets3d = (xs, ys, zs)

        for idx, finger in enumerate(finger_hierarchy):
            for i in range(len(finger) - 1):
                child_id = finger[i + 1]
                parent_id = finger[i]
                cx, cy, cz = frame_data[child_id]
                px, py, pz = frame_data[parent_id]
                lines[idx][i].set_data([px, cx], [py, cy])
                lines[idx][i].set_3d_properties([pz, cz])
        return scat, *[line for finger_lines in lines for line in finger_lines]

    # 播放动画
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
